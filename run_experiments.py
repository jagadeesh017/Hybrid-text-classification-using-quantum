#!/usr/bin/env python
"""
Comprehensive experiment runner for CQKSAN-DeBERTa project.

Trains all models (baseline, reduced, hybrid) and collects comparative results.
Generates results table and analysis for viva presentation.

Usage:
    python run_experiments.py --all
    python run_experiments.py --baseline
    python run_experiments.py --reduced
    python run_experiments.py --hybrid
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiments.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hqnlp import load_config
from hqnlp.data import build_dataloaders
from hqnlp.models import build_model
from hqnlp.training.trainer import Trainer


class ExperimentRunner:
    """Run experiments across different model configurations."""

    def __init__(self):
        self.results = {}
        self.experiment_dir = Path("results/experiments")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Experiments directory: {self.experiment_dir}")

    def run_experiment(self, config_path: str, model_name: str) -> dict:
        """Run single experiment.
        
        Args:
            config_path: Path to config YAML
            model_name: Name of model (baseline, reduced, hybrid)
            
        Returns:
            dict: Experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running {model_name.upper()} Model Experiment")
        logger.info(f"{'='*70}")
        
        try:
            start_time = time.time()
            
            # Load config
            logger.info(f"Loading config: {config_path}")
            config = load_config(config_path)
            
            # Build data loaders
            logger.info("Building data loaders...")
            train_loader, eval_loader, tokenizer, num_labels, label_names = build_dataloaders(
                config.data, config.model, config.training
            )
            config.model.num_labels = num_labels
            if not config.inference.class_names:
                config.inference.class_names = label_names
            
            logger.info(f"Label names: {label_names}")
            logger.info(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
            
            # Build model
            logger.info(f"Building {model_name} model...")
            model = build_model(config.model)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            # Train model
            logger.info("Starting training...")
            trainer = Trainer(config, model, train_loader, eval_loader, tokenizer, label_names)
            summary = trainer.fit()
            
            elapsed_time = time.time() - start_time
            
            # Process results
            result = {
                "model_name": model_name,
                "config_path": config_path,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "training_time_seconds": elapsed_time,
                "training_time_minutes": round(elapsed_time / 60, 2),
                "best_metric": summary["best_metric"],
                "final_eval": summary["final_eval"],
                "checkpoint_path": summary["output_dir"],
                "training_epochs": len(summary["history"]),
            }
            
            logger.info(f"\n{model_name.upper()} Experiment Results:")
            logger.info(f"  Best F1: {result['best_metric']:.4f}")
            logger.info(f"  Training time: {result['training_time_minutes']} minutes")
            logger.info(f"  Checkpoint: {result['checkpoint_path']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed for {model_name}: {e}", exc_info=True)
            return {"model_name": model_name, "error": str(e), "status": "FAILED"}

    def run_all_experiments(self) -> dict:
        """Run all experiments."""
        experiments = [
            ("configs/baseline.yaml", "baseline"),
            ("configs/reduced.yaml", "reduced"),
            ("configs/hybrid.yaml", "hybrid"),
        ]
        
        all_results = {}
        for config_path, model_name in experiments:
            result = self.run_experiment(config_path, model_name)
            all_results[model_name] = result
            
            # Save individual result
            result_file = self.experiment_dir / f"result_{model_name}_{self.timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved result to: {result_file}")
        
        return all_results

    def generate_comparison_table(self, results: dict) -> str:
        """Generate markdown comparison table.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            str: Markdown table
        """
        table = "# Experiment Results Comparison\n\n"
        table += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        table += f"**Dataset:** IMDB (2000 train, 400 eval samples)\n\n"
        
        table += "## Metrics Comparison\n\n"
        table += "| Model | Accuracy | Precision | Recall | F1-Score | Training Time (min) | Parameters |\n"
        table += "|-------|----------|-----------|--------|----------|-------------------|-------------|\n"
        
        for model_name, result in results.items():
            if "error" in result:
                table += f"| {model_name} | ERROR | - | - | - | - | - |\n"
                continue
            
            final_eval = result.get("final_eval", {})
            accuracy = final_eval.get("accuracy", 0)
            precision = final_eval.get("precision", 0)
            recall = final_eval.get("recall", 0)
            f1 = final_eval.get("f1", 0)
            time_min = result.get("training_time_minutes", 0)
            params = result.get("trainable_parameters", 0)
            
            table += f"| {model_name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {time_min:.1f} | {params:,} |\n"
        
        return table

    def generate_analysis(self, results: dict) -> str:
        """Generate analysis report.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            str: Markdown analysis
        """
        analysis = "## Detailed Analysis\n\n"
        
        # Extract F1 scores for comparison
        baseline_f1 = results.get("baseline", {}).get("best_metric", 0)
        reduced_f1 = results.get("reduced", {}).get("best_metric", 0)
        hybrid_f1 = results.get("hybrid", {}).get("best_metric", 0)
        
        if baseline_f1 > 0:
            reduced_improvement = ((reduced_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            hybrid_improvement = ((hybrid_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            quantum_benefit = ((hybrid_f1 - reduced_f1) / reduced_f1 * 100) if reduced_f1 > 0 else 0
            
            analysis += f"### Performance Improvements\n\n"
            analysis += f"- **Baseline F1:** {baseline_f1:.4f} (reference)\n"
            analysis += f"- **Reduced F1:** {reduced_f1:.4f} ({reduced_improvement:+.1f}% vs baseline)\n"
            analysis += f"- **Hybrid F1:** {hybrid_f1:.4f} ({hybrid_improvement:+.1f}% vs baseline)\n"
            analysis += f"- **Quantum Benefit:** {quantum_benefit:+.1f}% (hybrid vs reduced)\n\n"
            
            # Best model
            best_model = max([
                ("baseline", baseline_f1),
                ("reduced", reduced_f1),
                ("hybrid", hybrid_f1)
            ], key=lambda x: x[1])
            analysis += f"**Best Model:** {best_model[0].upper()} with F1={best_model[1]:.4f}\n\n"
        
        # Training times
        analysis += "### Training Efficiency\n\n"
        for model_name, result in results.items():
            if "error" not in result:
                time_min = result.get("training_time_minutes", 0)
                params = result.get("trainable_parameters", 0)
                if params > 0:
                    efficiency = (result.get("best_metric", 0) / time_min * 100) if time_min > 0 else 0
                    analysis += f"- **{model_name}:** {time_min:.1f} min | {params:,} params\n"
        
        analysis += "\n### Architecture Insights\n\n"
        analysis += "- **Baseline:** Simple DeBERTa + classifier (reference model)\n"
        analysis += "- **Reduced:** Adds feature reduction layer (dims 768→128→4)\n"
        analysis += "- **Hybrid:** Integrates quantum-inspired kernel attention + fusion module\n"
        
        if quantum_benefit > 0:
            analysis += f"\n✅ **Quantum advantage demonstrated:** Hybrid provides {quantum_benefit:.1f}% improvement over classical reduced model\n"
        else:
            analysis += f"\n⚠️ **Note:** Hybrid model shows {quantum_benefit:.1f}% vs reduced (may indicate classical efficiency or limited quantum benefit in simulation)\n"
        
        return analysis

    def save_results(self, results: dict):
        """Save all results to file.
        
        Args:
            results: Dictionary of model results
        """
        # Save full results
        results_file = self.experiment_dir / f"all_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved all results to: {results_file}")
        
        # Generate and save markdown report
        report = "# CQKSAN-DeBERTa Hybrid Quantum NLP - Experiment Results\n\n"
        report += f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Python Version:** {sys.version.split()[0]}\n"
        report += f"**PyTorch Version:** {torch.__version__}\n"
        report += f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n"
        
        report += self.generate_comparison_table(results)
        report += "\n\n"
        report += self.generate_analysis(results)
        
        report_file = self.experiment_dir / f"report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to: {report_file}")
        
        # Also save to standard results location
        summary_file = Path("results/RESULTS.md")
        with open(summary_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved summary to: {summary_file}")

    def print_summary(self, results: dict):
        """Print summary to console.
        
        Args:
            results: Dictionary of model results
        """
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print("\n" + self.generate_comparison_table(results))
        print("\n" + self.generate_analysis(results))
        
        print("\n" + "="*80)
        print("All results saved to: results/experiments/")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run CQKSAN-DeBERTa experiments.")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument("--reduced", action="store_true", help="Run reduced only")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid only")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer samples")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    try:
        if args.all or not any([args.baseline, args.reduced, args.hybrid]):
            results = runner.run_all_experiments()
        else:
            results = {}
            if args.baseline:
                results["baseline"] = runner.run_experiment("configs/baseline.yaml", "baseline")
            if args.reduced:
                results["reduced"] = runner.run_experiment("configs/reduced.yaml", "reduced")
            if args.hybrid:
                results["hybrid"] = runner.run_experiment("configs/hybrid.yaml", "hybrid")
        
        runner.save_results(results)
        runner.print_summary(results)
        
    except Exception as e:
        logger.error(f"Experiment runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
