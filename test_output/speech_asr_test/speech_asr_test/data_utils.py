"""
Data processing utilities using Polars for high-performance data manipulation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_and_process_dataset(
    dataset_name: str,
    subset_size: Optional[int] = None,
    text_column: str = "text",
    label_column: str = "label"
) -> DatasetDict:
    """
    Load and process a dataset using Polars for efficient data manipulation.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        subset_size: Optional size to limit dataset for faster iteration
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Processed DatasetDict ready for training
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    processed_splits = {}
    
    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split with {len(split_data)} examples")
        
        # Convert to Polars DataFrame for efficient processing
        df = pl.DataFrame({
            text_column: split_data[text_column],
            label_column: split_data[label_column]
        })
        
        # Data cleaning with Polars
        df = (
            df
            .filter(pl.col(text_column).str.len_chars() > 10)  # Remove very short texts
            .filter(pl.col(text_column).str.len_chars() < 1000)  # Remove very long texts
            .filter(pl.col(text_column).is_not_null())  # Remove null texts
            .unique(subset=[text_column])  # Remove duplicates
        )
        
        # Apply subset if requested
        if subset_size and len(df) > subset_size:
            # Stratified sampling to maintain label distribution
            df_sampled = (
                df
                .group_by(label_column)
                .sample(n=subset_size // df[label_column].n_unique(), seed=42)
                .sort(label_column)
            )
            df = df_sampled
            logger.info(f"Sampled {len(df)} examples from {split_name}")
        
        # Convert back to HuggingFace Dataset
        processed_splits[split_name] = Dataset.from_dict({
            text_column: df[text_column].to_list(),
            label_column: df[label_column].to_list()
        })
        
        logger.info(f"Processed {split_name}: {len(processed_splits[split_name])} examples")
    
    return DatasetDict(processed_splits)


def analyze_dataset_stats(dataset: DatasetDict, text_column: str = "text") -> Dict:
    """
    Analyze dataset statistics using Polars for fast computation.
    
    Args:
        dataset: HuggingFace DatasetDict
        text_column: Name of the text column
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    
    for split_name, split_data in dataset.items():
        # Convert to Polars for analysis
        df = pl.DataFrame({
            text_column: split_data[text_column],
            "label": split_data["label"]
        })
        
        # Compute statistics
        text_lengths = df.select(pl.col(text_column).str.len_chars().alias("length"))
        
        split_stats = {
            "num_examples": len(df),
            "num_labels": df["label"].n_unique(),
            "label_distribution": df["label"].value_counts().sort("label").to_dict(as_series=False),
            "text_length_stats": {
                "mean": float(text_lengths["length"].mean()),
                "median": float(text_lengths["length"].median()),
                "min": int(text_lengths["length"].min()),
                "max": int(text_lengths["length"].max()),
                "std": float(text_lengths["length"].std())
            },
            "avg_words_per_text": float(
                df.select(pl.col(text_column).str.split(" ").list.len().mean()).item()
            )
        }
        
        stats[split_name] = split_stats
        
        logger.info(f"{split_name} statistics:")
        logger.info(f"  Examples: {split_stats['num_examples']}")
        logger.info(f"  Labels: {split_stats['num_labels']}")
        logger.info(f"  Avg text length: {split_stats['text_length_stats']['mean']:.1f} chars")
        logger.info(f"  Avg words per text: {split_stats['avg_words_per_text']:.1f}")
    
    return stats


def create_data_summary_report(dataset: DatasetDict, output_path: Optional[Path] = None) -> str:
    """
    Create a comprehensive data summary report using Polars analysis.
    
    Args:
        dataset: HuggingFace DatasetDict
        output_path: Optional path to save the report
        
    Returns:
        String containing the formatted report
    """
    stats = analyze_dataset_stats(dataset)
    
    report_lines = [
        "# Dataset Summary Report",
        "",
        f"Generated for dataset with {len(dataset)} splits",
        ""
    ]
    
    for split_name, split_stats in stats.items():
        report_lines.extend([
            f"## {split_name.title()} Split",
            "",
            f"- **Examples**: {split_stats['num_examples']:,}",
            f"- **Unique Labels**: {split_stats['num_labels']}",
            "",
            "### Text Length Statistics",
            f"- Mean: {split_stats['text_length_stats']['mean']:.1f} characters",
            f"- Median: {split_stats['text_length_stats']['median']:.1f} characters", 
            f"- Range: {split_stats['text_length_stats']['min']}-{split_stats['text_length_stats']['max']} characters",
            f"- Standard Deviation: {split_stats['text_length_stats']['std']:.1f}",
            "",
            f"### Average Words per Text: {split_stats['avg_words_per_text']:.1f}",
            "",
            "### Label Distribution",
        ])
        
        # Add label distribution
        label_dist = split_stats['label_distribution']
        for i, (label, count) in enumerate(zip(label_dist['label'], label_dist['count'])):
            percentage = (count / split_stats['num_examples']) * 100
            report_lines.append(f"- Label {label}: {count:,} ({percentage:.1f}%)")
        
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if output_path:
        output_path.write_text(report)
        logger.info(f"Data summary report saved to {output_path}")
    
    return report


def export_sample_data(
    dataset: DatasetDict, 
    num_samples: int = 10,
    output_dir: Optional[Path] = None
) -> None:
    """
    Export sample data to files using Polars for efficient processing.
    
    Args:
        dataset: HuggingFace DatasetDict
        num_samples: Number of samples to export per split
        output_dir: Directory to save sample files
    """
    if output_dir is None:
        output_dir = Path("data_samples")
    
    output_dir.mkdir(exist_ok=True)
    
    for split_name, split_data in dataset.items():
        # Convert to Polars and sample
        df = pl.DataFrame({
            "text": split_data["text"],
            "label": split_data["label"]
        })
        
        # Sample data
        sample_df = df.sample(n=min(num_samples, len(df)), seed=42)
        
        # Save as CSV
        output_path = output_dir / f"{split_name}_samples.csv"
        sample_df.write_csv(output_path)
        
        logger.info(f"Exported {len(sample_df)} {split_name} samples to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load and process IMDB dataset
    dataset = load_and_process_dataset("imdb", subset_size=1000)
    
    # Analyze statistics
    stats = analyze_dataset_stats(dataset)
    
    # Create report
    report = create_data_summary_report(dataset, Path("data_report.md"))
    print(report)
    
    # Export samples
    export_sample_data(dataset, num_samples=5)