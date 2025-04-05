import os
import time
import click
import torch
import wandb
import numpy as np
import psutil
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from transformers import ViTImageProcessorFast, ViTConfig, AutoConfig, BitsAndBytesConfig
from benchmark.utils import estimate_flops
from common.wandb_logger import init_wandb_run
from common.metrics import recall_at_k
from common.logger import logger
from common.config import BenchmarkConfig, TrainingConfig
from models.utils import compute_similarity_scores
from datasets.lfw.dataset import LFWBenchmarkDataset
from models.vit_encoder import VitEncoder, ExtendedViTConfig

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
    model.eval()
    
    # dataset = LFWBenchmarkDataset(
    #     dir_path=lfw_dataset_dir,
    #     processor=processor,
    #     max_images_per_identity=max_images_per_identity
    # )
    # dataloader = DataLoader(dataset, batch_size=training_config.test_batch_size, shuffle=False, num_workers=training_config.num_workers)
    
    # logger.info(f"Loaded dataset with {len(dataset)} images")
    
    # # Compute embeddings for all images
    # logger.info("Computing embeddings for all images...")
    # all_embeddings = []
    # all_image_paths = []
    # all_identities = []
    
    # with torch.no_grad():
    #     for batch in tqdm(dataloader):
    #         images, image_paths, identities = batch
    #         images = images.to(device)
    #         embeddings = model(images)
    #         all_embeddings.append(embeddings.cpu().numpy())
    #         all_image_paths.extend(image_paths)
    #         all_identities.extend(identities)
    
    # all_embeddings = np.vstack(all_embeddings)
    # vector_dim = all_embeddings.shape[1]
    
    # # For each identity, use one image as query and the rest as gallery
    # logger.info("Evaluating model...")
    # identity_to_indices = {}
    # for i, identity in enumerate(all_identities):
    #     if identity not in identity_to_indices:
    #         identity_to_indices[identity] = []
    #     identity_to_indices[identity].append(i)
    
    # # Keep only identities with multiple images
    # identity_to_indices = {k: v for k, v in identity_to_indices.items() if len(v) > 1}
    
    # # Evaluate with full reranking
    # recalls = {1: [], 10: [], 100: [], 1000: []}
    
    # # Measure similarity computation time and FLOPs
    # similarity_start_time = time.time()
    # cpu_sim_start = psutil.cpu_times()
    # total_similarity_flops = 0
    # total_queries = 0
    
    # for identity, indices in tqdm(identity_to_indices.items(), desc="Evaluating"):
    #     # Use first image as query
    #     query_idx = indices[0]
    #     query_embedding = all_embeddings[query_idx:query_idx+1]
    #     total_queries += 1
        
    #     # Calculate similarity scores with all other images
    #     similarity_scores = compute_similarity_scores(query_embedding, all_embeddings)
        
    #     # Estimate FLOPs for this query - dot product with all gallery images
    #     query_flops = estimate_flops(len(all_embeddings), vector_dim, 2)
    #     total_similarity_flops += query_flops
        
    #     # Sort by similarity, excluding the query itself
    #     sorted_indices = np.argsort(-similarity_scores)
    #     sorted_indices = [idx for idx in sorted_indices if idx != query_idx][:1000]
        
    #     # Calculate recall metrics
    #     for k in recalls.keys():
    #         hit = any(idx in sorted_indices[:k] for idx in indices if idx != query_idx)
    #         recalls[k].append(1 if hit else 0)
    
    # similarity_duration = time.time() - similarity_start_time
    # cpu_sim_end = psutil.cpu_times()
    # cpu_sim_user_time = cpu_sim_end.user - cpu_sim_start.user
    # cpu_sim_system_time = cpu_sim_end.system - cpu_sim_start.system
    
    # avg_flops_per_query = total_similarity_flops / total_queries if total_queries > 0 else 0
    
    # logger.info(f"Similarity computation completed in {similarity_duration:.2f} seconds")
    # logger.info(f"CPU time for similarity - User: {cpu_sim_user_time:.2f}s, System: {cpu_sim_system_time:.2f}s")
    # logger.info(f"Average FLOPs per query: {avg_flops_per_query:,}")
    
    # # Log results
    # results = {
    #     f"recall@{k}": np.mean(recalls[k]) * 100 for k in recalls.keys()
    # }
    
    # # Add performance metrics
    # performance_metrics = {
    #     "similarity_computation_duration": similarity_duration,
    #     "similarity_cpu_user_time": cpu_sim_user_time,
    #     "similarity_cpu_system_time": cpu_sim_system_time,
    #     "average_flops_per_query": avg_flops_per_query,
    #     "total_queries": total_queries
    # }
    
    # results.update(performance_metrics)
    # wandb_run.log(results)
    
    # for k, value in results.items():
    #     if isinstance(value, float):
    #         logger.info(f"{k}: {value:.2f}")
    #     else:
    #         logger.info(f"{k}: {value}")
    
    
    # benchmark_duration = time.time() - start_time
    # logger.info(f"Benchmark completed in {benchmark_duration:.2f} seconds")
    # wandb_run.summary.update({
    #     "benchmark_duration": benchmark_duration,
    #     **performance_metrics
    # })


if __name__ == "__main__":
    full_rerank_benchmark()