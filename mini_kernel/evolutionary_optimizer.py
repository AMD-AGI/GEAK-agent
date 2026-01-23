#!/usr/bin/env python3
"""
Evolutionary Optimizer Component

Integrates OpenEvolve-style evolutionary optimization:
1. Evaluate initial strategies (population)
2. Select best performers (selection)
3. Combine successful strategies (crossover)
4. Mutate to explore variations (mutation)
5. Iterate until convergence or budget exhausted

This creates hybrid strategies that combine multiple optimizations
for potentially higher speedups than any single strategy.
"""

import random
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from pathlib import Path


class OptimizationType(Enum):
    """Types of optimizations that can be combined."""
    PYTORCH_REPLACEMENT = "pytorch_replacement"
    HIP_GRAPH = "hip_graph"
    PREFILL_CONSTANTS = "prefill_constants"
    ELIMINATE_REDUNDANT = "eliminate_redundant"
    VECTORIZED_COPY = "vectorized_copy"
    CONTIGUOUS_BUFFERS = "contiguous_buffers"
    FUSED_OPERATIONS = "fused_operations"
    ASYNC_COPY = "async_copy"


@dataclass
class Genome:
    """
    A genome represents a combination of optimizations.
    
    Each gene is a boolean indicating whether an optimization is enabled.
    """
    genes: Dict[OptimizationType, bool] = field(default_factory=dict)
    fitness: float = 0.0  # Speedup (higher is better)
    latency_us: float = float('inf')
    is_correct: bool = False
    generation: int = 0
    
    def __post_init__(self):
        # Initialize all genes to False if empty
        if not self.genes:
            for opt in OptimizationType:
                self.genes[opt] = False
    
    def enabled_optimizations(self) -> List[OptimizationType]:
        """Get list of enabled optimizations."""
        return [opt for opt, enabled in self.genes.items() if enabled]
    
    def to_strategy_name(self) -> str:
        """Generate a name for this genome's strategy."""
        enabled = self.enabled_optimizations()
        if not enabled:
            return "baseline"
        return "_".join(sorted([opt.value for opt in enabled]))
    
    def copy(self) -> 'Genome':
        """Create a deep copy."""
        new = Genome()
        new.genes = copy.deepcopy(self.genes)
        new.fitness = self.fitness
        new.latency_us = self.latency_us
        new.is_correct = self.is_correct
        new.generation = self.generation
        return new


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""
    population_size: int = 8
    generations: int = 5
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elite_count: int = 2  # Top N to preserve unchanged
    tournament_size: int = 3
    
    # Compatibility matrix: which optimizations can be combined
    # Some optimizations conflict (e.g., can't use both pytorch_replacement AND hip_graph on same operation)
    compatible_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    
    def __post_init__(self):
        # Define compatible optimization pairs
        self.compatible_pairs = {
            # HIP Graph works with most things
            ("hip_graph", "prefill_constants"),
            ("hip_graph", "eliminate_redundant"),
            ("hip_graph", "contiguous_buffers"),
            # Prefill works with many
            ("prefill_constants", "eliminate_redundant"),
            ("prefill_constants", "vectorized_copy"),
            ("prefill_constants", "contiguous_buffers"),
            # Others
            ("vectorized_copy", "contiguous_buffers"),
            ("eliminate_redundant", "contiguous_buffers"),
        }


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer that combines successful strategies.
    
    Uses genetic algorithm principles:
    - Population of strategy combinations (genomes)
    - Selection based on fitness (speedup)
    - Crossover to combine successful strategies
    - Mutation to explore new combinations
    """
    
    def __init__(self, config: EvolutionConfig, 
                 evaluate_fn, 
                 baseline_latency_us: float):
        """
        Args:
            config: Evolution configuration
            evaluate_fn: Function(genome) -> (latency_us, is_correct)
            baseline_latency_us: Baseline latency for fitness calculation
        """
        self.config = config
        self.evaluate = evaluate_fn
        self.baseline_latency_us = baseline_latency_us
        
        self.population: List[Genome] = []
        self.best_genome: Optional[Genome] = None
        self.history: List[Dict[str, Any]] = []
    
    def run(self) -> Genome:
        """Run the evolutionary optimization loop."""
        print("\n" + "=" * 60)
        print("  EVOLUTIONARY OPTIMIZATION")
        print("=" * 60)
        
        # Initialize population with known good strategies
        self._initialize_population()
        
        # Evaluate initial population
        self._evaluate_population()
        
        # Evolution loop
        for gen in range(self.config.generations):
            print(f"\n--- Generation {gen + 1}/{self.config.generations} ---")
            
            # Select parents
            parents = self._selection()
            
            # Create offspring through crossover
            offspring = self._crossover(parents)
            
            # Mutate offspring
            offspring = self._mutation(offspring)
            
            # Update generation number
            for genome in offspring:
                genome.generation = gen + 1
            
            # Evaluate offspring
            for genome in offspring:
                latency, correct = self.evaluate(genome)
                genome.latency_us = latency
                genome.is_correct = correct
                genome.fitness = self._calculate_fitness(latency, correct)
            
            # Elitism: keep best from previous generation
            elites = sorted(self.population, key=lambda g: g.fitness, reverse=True)
            elites = [g for g in elites if g.is_correct][:self.config.elite_count]
            
            # Replace population
            self.population = elites + offspring
            self.population = sorted(self.population, key=lambda g: g.fitness, reverse=True)
            self.population = self.population[:self.config.population_size]
            
            # Update best
            valid = [g for g in self.population if g.is_correct]
            if valid:
                best = max(valid, key=lambda g: g.fitness)
                if self.best_genome is None or best.fitness > self.best_genome.fitness:
                    self.best_genome = best.copy()
                    print(f"  ⭐ NEW BEST: {best.to_strategy_name()}")
                    print(f"     Latency: {best.latency_us:.2f} μs")
                    print(f"     Speedup: {best.fitness:.2f}x")
            
            # Record history
            self.history.append({
                "generation": gen + 1,
                "best_fitness": self.best_genome.fitness if self.best_genome else 0,
                "best_latency": self.best_genome.latency_us if self.best_genome else 0,
                "population_size": len(self.population),
                "valid_count": len(valid),
            })
            
            self._print_generation_summary()
        
        print("\n" + "=" * 60)
        print("  EVOLUTION COMPLETE")
        print("=" * 60)
        if self.best_genome:
            print(f"  Best Strategy: {self.best_genome.to_strategy_name()}")
            print(f"  Best Latency:  {self.best_genome.latency_us:.2f} μs")
            print(f"  Best Speedup:  {self.best_genome.fitness:.2f}x")
            print(f"  Optimizations: {[o.value for o in self.best_genome.enabled_optimizations()]}")
        
        return self.best_genome
    
    def _initialize_population(self):
        """Initialize population with known strategies + random combinations."""
        print("  Initializing population...")
        
        # Add single-optimization genomes (known strategies)
        single_opts = [
            [OptimizationType.HIP_GRAPH],
            [OptimizationType.PREFILL_CONSTANTS],
            [OptimizationType.PYTORCH_REPLACEMENT],
            [OptimizationType.ELIMINATE_REDUNDANT],
        ]
        
        for opts in single_opts:
            genome = Genome()
            for opt in opts:
                genome.genes[opt] = True
            self.population.append(genome)
        
        # Add known good combinations
        good_combos = [
            [OptimizationType.HIP_GRAPH, OptimizationType.PREFILL_CONSTANTS],
            [OptimizationType.PREFILL_CONSTANTS, OptimizationType.ELIMINATE_REDUNDANT],
        ]
        
        for opts in good_combos:
            genome = Genome()
            for opt in opts:
                genome.genes[opt] = True
            self.population.append(genome)
        
        # Fill rest with random combinations
        while len(self.population) < self.config.population_size:
            genome = Genome()
            # Randomly enable 1-3 optimizations
            num_opts = random.randint(1, 3)
            opts = random.sample(list(OptimizationType), num_opts)
            for opt in opts:
                genome.genes[opt] = True
            
            # Check if valid (compatible)
            if self._is_compatible(genome):
                self.population.append(genome)
        
        print(f"  Population size: {len(self.population)}")
    
    def _evaluate_population(self):
        """Evaluate fitness of all genomes in population."""
        print("  Evaluating initial population...")
        
        for i, genome in enumerate(self.population):
            print(f"    [{i+1}/{len(self.population)}] {genome.to_strategy_name()}")
            latency, correct = self.evaluate(genome)
            genome.latency_us = latency
            genome.is_correct = correct
            genome.fitness = self._calculate_fitness(latency, correct)
            
            status = "✓" if correct else "✗"
            print(f"         {status} Latency: {latency:.2f} μs, Fitness: {genome.fitness:.2f}")
        
        # Update best
        valid = [g for g in self.population if g.is_correct]
        if valid:
            self.best_genome = max(valid, key=lambda g: g.fitness).copy()
    
    def _calculate_fitness(self, latency_us: float, is_correct: bool) -> float:
        """Calculate fitness score (speedup over baseline)."""
        if not is_correct or latency_us <= 0:
            return 0.0
        return self.baseline_latency_us / latency_us
    
    def _selection(self) -> List[Genome]:
        """Select parents using tournament selection."""
        parents = []
        valid = [g for g in self.population if g.is_correct]
        
        if len(valid) < 2:
            return valid
        
        num_parents = max(2, len(valid) // 2)
        
        for _ in range(num_parents):
            # Tournament selection
            tournament = random.sample(valid, min(self.config.tournament_size, len(valid)))
            winner = max(tournament, key=lambda g: g.fitness)
            parents.append(winner.copy())
        
        return parents
    
    def _crossover(self, parents: List[Genome]) -> List[Genome]:
        """Create offspring by combining parent genes."""
        offspring = []
        
        if len(parents) < 2:
            return [p.copy() for p in parents]
        
        # Create pairs
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            
            if random.random() < self.config.crossover_rate:
                # Uniform crossover
                child1 = Genome()
                child2 = Genome()
                
                for opt in OptimizationType:
                    if random.random() < 0.5:
                        child1.genes[opt] = p1.genes[opt]
                        child2.genes[opt] = p2.genes[opt]
                    else:
                        child1.genes[opt] = p2.genes[opt]
                        child2.genes[opt] = p1.genes[opt]
                
                # Only keep compatible children
                if self._is_compatible(child1):
                    offspring.append(child1)
                if self._is_compatible(child2):
                    offspring.append(child2)
            else:
                offspring.append(p1.copy())
                offspring.append(p2.copy())
        
        return offspring
    
    def _mutation(self, genomes: List[Genome]) -> List[Genome]:
        """Mutate genomes by flipping random genes."""
        for genome in genomes:
            if random.random() < self.config.mutation_rate:
                # Flip a random gene
                opt = random.choice(list(OptimizationType))
                genome.genes[opt] = not genome.genes[opt]
                
                # Ensure still compatible
                if not self._is_compatible(genome):
                    genome.genes[opt] = not genome.genes[opt]
        
        return genomes
    
    def _is_compatible(self, genome: Genome) -> bool:
        """Check if enabled optimizations are compatible."""
        enabled = genome.enabled_optimizations()
        
        if len(enabled) <= 1:
            return True
        
        # Check all pairs
        for i, opt1 in enumerate(enabled):
            for opt2 in enabled[i+1:]:
                pair = tuple(sorted([opt1.value, opt2.value]))
                if pair not in self.config.compatible_pairs:
                    # Check reverse
                    reverse_pair = (pair[1], pair[0])
                    if reverse_pair not in self.config.compatible_pairs:
                        return False
        
        return True
    
    def _print_generation_summary(self):
        """Print summary of current generation."""
        valid = [g for g in self.population if g.is_correct]
        
        print(f"  Population: {len(self.population)} | Valid: {len(valid)}")
        
        if valid:
            best = max(valid, key=lambda g: g.fitness)
            avg_fitness = sum(g.fitness for g in valid) / len(valid)
            print(f"  Best: {best.fitness:.2f}x | Avg: {avg_fitness:.2f}x")


def generate_hybrid_code(genome: Genome, base_harness: str) -> str:
    """
    Generate optimized code that combines multiple optimizations.
    
    This is the key function that creates hybrid strategies by
    combining code from multiple optimization techniques.
    """
    enabled = genome.enabled_optimizations()
    
    code_parts = []
    code_parts.append('''
import sys
sys.path.insert(0, "/workspace")
from test_harness import *
''')
    
    # Pre-initialization (runs once)
    pre_init = []
    
    if OptimizationType.PREFILL_CONSTANTS in enabled:
        pre_init.append("# Pre-fill constant values")
        pre_init.append("ids_out[:, topk] = 256")
        pre_init.append("w_out[:, topk] = 0.0")
    
    if OptimizationType.CONTIGUOUS_BUFFERS in enabled:
        pre_init.append("# Ensure contiguous")
        pre_init.append("ids_out = ids_out.contiguous()")
        pre_init.append("w_out = w_out.contiguous()")
    
    if pre_init:
        code_parts.append("\n".join(pre_init))
    
    # Build pipeline function
    pipeline_code = []
    pipeline_code.append("def pipeline():")
    pipeline_code.append("    biased_grouped_topk_hip(gating, bias, topk_weights, topk_ids,")
    pipeline_code.append("                           num_expert_group, topk_group, True, 1.0)")
    
    if OptimizationType.VECTORIZED_COPY in enabled:
        pipeline_code.append("    ids_out[:, :topk].copy_(topk_ids)")
        pipeline_code.append("    w_out[:, :topk].copy_(topk_weights)")
    else:
        pipeline_code.append("    ids_out[:, :topk] = topk_ids")
        pipeline_code.append("    w_out[:, :topk] = topk_weights")
    
    if OptimizationType.PREFILL_CONSTANTS not in enabled:
        pipeline_code.append("    ids_out[:, topk] = 256")
        pipeline_code.append("    w_out[:, topk] = 0.0")
    
    pipeline_code.append("    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp,")
    pipeline_code.append("                   num_valid, moe_buf, total_experts, unit_size)")
    
    code_parts.append("\n".join(pipeline_code))
    
    # HIP Graph capture if enabled
    if OptimizationType.HIP_GRAPH in enabled:
        code_parts.append('''
# Warmup for graph capture
moe_buf.zero_()
for _ in range(100):
    pipeline()
torch.cuda.synchronize()

# Capture HIP Graph
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    graph = torch.cuda.CUDAGraph()
    moe_buf.zero_()
    with torch.cuda.graph(graph):
        pipeline()

def run_optimized():
    moe_buf.zero_()
    graph.replay()
''')
    else:
        code_parts.append('''
def run_optimized():
    moe_buf.zero_()
    pipeline()
''')
    
    # Verification and benchmarking
    code_parts.append('''
# Verify and benchmark
run_optimized()
torch.cuda.synchronize()
correct = check_correctness()

# Benchmark
for _ in range(1000):
    run_optimized()
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(3000):
    run_optimized()
end.record()
torch.cuda.synchronize()
latency = start.elapsed_time(end) * 1000 / 3000

with open("/workspace/opt_result.json", "w") as f:
    json.dump({
        "correct": correct["passed"],
        "mismatches": correct.get("mismatches", []),
        "latency_us": latency,
        "strategy": "''' + genome.to_strategy_name() + '''",
        "optimizations": ''' + str([o.value for o in enabled]) + '''
    }, f)
''')
    
    return "\n".join(code_parts)


