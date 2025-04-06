import os
import time
import click
import torch
import wandb
import psutil
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from transformers import ViTImageProcessorFast, ViTConfig, AutoConfig, BitsAndBytesConfig
from benchmark.utils import partition_lfw_images
from common.wandb_logger import init_wandb_run
from common.metrics import recall_at_k
from common.logger import logger
from common.config import BenchmarkConfig, TrainingConfig
from datasets.lfw.dataset import LFWBenchmarkDataset
from models.utils import compute_similarity_scores
from models.vit_encoder import VitEncoder, ExtendedViTConfig
from training.scoring import maxsim
from einops import rearrange

@click.command()
@click.option("--model-dir", required=True, help="Path to model checkpoint directory")
@click.option("--lfw-dataset-dir", required=True, help="Path to LFW dataset directory")
@click.option("--wandb-run-id", required=True, help="W&B run ID to load the training config from")
@click.option("--max-images-per-identity", default=2, help="Maximum number of images per identity")
def full_rerank_benchmark(
    model_dir: str,
    lfw_dataset_dir: str,
    wandb_run_id: str,
    max_images_per_identity: int
):
    """Full reranking benchmark using cross-encoding style.
    
    This benchmark compares each query against the entire gallery,
    without using a vector index for pre-filtering.
    """
    
    logger.info("Starting full reranking benchmark...")
    start_time = time.time()
    
    wandb_run = init_wandb_run()
    benchmark_config = BenchmarkConfig(
        model_dir=model_dir,
        lfw_dataset_dir=lfw_dataset_dir,
        max_images_per_identity=max_images_per_identity,
        benchmark_type="full_rerank"
    )
    wandb_run.config.update(benchmark_config.model_dump())
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    training_config = TrainingConfig.load_from_wandb(wandb_run_id)
    finetuning_config = training_config.get_finetuning_config()
    model_config = training_config.get_model_config()
    vit_config: ViTConfig = AutoConfig.from_pretrained(model_config.pretrained_vit_name)
    extended_vit_config = ExtendedViTConfig(
        **vit_config.to_dict(), model_config=model_config
    )
    processor = ViTImageProcessorFast.from_pretrained(model_config.pretrained_vit_name)
    model = VitEncoder.from_pretrained(
        model_config.pretrained_vit_name,
        config=extended_vit_config,
        quantization_config=quantization_config,
    )
    model.load_from_checkpoint(model_dir)
    
    logger.info("Partitioning LFW images...")
    all_identities, all_image_paths, all_image_path_to_identity, test_identities, test_image_paths, test_image_path_to_identity = partition_lfw_images(
        lfw_dataset_dir, max_images_per_identity
    )
    
    logger.info("Loading LFW dataset...")
    full_dataset = LFWBenchmarkDataset(
        dir_path=lfw_dataset_dir,
        processor=processor,
        image_identities=all_identities,
        image_paths=all_image_paths
    )
    test_dataset = LFWBenchmarkDataset(
        dir_path=lfw_dataset_dir,
        processor=processor,
        image_identities=test_identities,
        image_paths=test_image_paths
    )
    
    logger.info("Initializing dataloaders...")
    full_dataloader = DataLoader(full_dataset, batch_size=training_config.test_batch_size, shuffle=False, num_workers=training_config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=training_config.test_batch_size, shuffle=False, num_workers=training_config.num_workers)
    
    logger.info(f"Loaded dataset with {len(full_dataset)} images")
    
    # Compute embeddings for all images
    logger.info("Computing embeddings for all images...")
    
    accelerator = Accelerator(mixed_precision=finetuning_config.precision)
    model, full_dataloader, test_dataloader = accelerator.prepare(model, full_dataloader, test_dataloader)
    all_embeddings = torch.tensor([], device=accelerator.device)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(full_dataloader):
            with accelerator.autocast():
                pixel_values, image_paths, identities = batch
                batch_size, num_images = pixel_values.shape[:2]
                images = rearrange(
                    pixel_values,
                    "batch_size num_images channel height width -> (batch_size num_images) channel height width",
                )
                embeddings = model(images)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                embeddings = rearrange(
                    embeddings,
                    "(batch_size num_images) seq_len reduced_dim -> batch_size num_images seq_len reduced_dim",
                    batch_size=batch_size,
                    num_images=num_images,
                )
                all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)
            

    similarity_start_time = time.time()
    cpu_sim_start = psutil.cpu_times()
    
    recalls = {1: [], 10: [], 100: [], 1000: []}
    max_k = max(recalls.keys())
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            with accelerator.autocast():
                pixel_values, image_paths, identities = batch
                batch_size, num_images = pixel_values.shape[:2]
                logger.info(f"Batch size: {batch_size}")
                logger.info(f"Num images: {num_images}")
                query_images = rearrange(
                    pixel_values,
                    "batch_size num_images channel height width -> (batch_size num_images) channel height width",
                )
                query_embeddings = model(query_images)
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
                query_embeddings = rearrange(
                    query_embeddings,
                    "(batch_size num_images) seq_len reduced_dim -> batch_size num_images seq_len reduced_dim",
                    batch_size=batch_size,
                    num_images=num_images,
                )
                                   
                for idx in range(batch_size):
                    query_embedding = query_embeddings[idx].unsqueeze(0)
                    query_path = image_paths[idx]
                    query_identity = identities[idx]
                    similarity_scores = maxsim(query_embedding, all_embeddings) # (batch_size, num_images)
                    similarity_scores = similarity_scores.squeeze(0)
                    
                    top_k_indices = torch.argsort(similarity_scores, descending=True)[:max_k]
                    top_k_indices = top_k_indices.cpu().tolist()

                    top_k_paths = [all_image_paths[i] for i in top_k_indices]
                    top_k_paths = [path for path in top_k_paths if path != query_path]
                    top_k_identities = [all_image_path_to_identity[path] for path in top_k_paths]
                    logger.info(f"Target identity: {query_identity}")
                    logger.info(f"Top k identities: {top_k_identities}")
                    for k in recalls.keys():
                        hit = any(identity == query_identity for identity in top_k_identities[:k])
                        recalls[k].append(1 if hit else 0)
                    
    similarity_duration = time.time() - similarity_start_time
    cpu_sim_end = psutil.cpu_times()
    cpu_sim_user_time = cpu_sim_end.user - cpu_sim_start.user
    cpu_sim_system_time = cpu_sim_end.system - cpu_sim_start.system
    
    logger.info(f"Similarity computation completed in {similarity_duration:.2f} seconds")
    logger.info(f"CPU time for similarity - User: {cpu_sim_user_time:.2f}s, System: {cpu_sim_system_time:.2f}s")
    
    # Log results
    results = {
        f"recall@{k}": float(torch.mean(torch.tensor(recalls[k], dtype=torch.float)) * 100) for k in recalls.keys()
    }
    
    # Add performance metrics
    performance_metrics = {
        "similarity_computation_duration": similarity_duration,
        "similarity_cpu_user_time": cpu_sim_user_time,
        "similarity_cpu_system_time": cpu_sim_system_time
    }
    
    results.update(performance_metrics)
    wandb_run.log(results)
    
    for k, value in results.items():
        if isinstance(value, float):
            logger.info(f"{k}: {value:.2f}")
        else:
            logger.info(f"{k}: {value}")
    
    
    benchmark_duration = time.time() - start_time
    logger.info(f"Benchmark completed in {benchmark_duration:.2f} seconds")
    wandb_run.summary.update({
        "benchmark_duration": benchmark_duration,
        **performance_metrics
    })


if __name__ == "__main__":
    full_rerank_benchmark()