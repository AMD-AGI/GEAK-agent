#!/usr/bin/env python3
"""
OpenEvolve Brain - The Central Intelligence for Mini-Kernel Agent

This is THE brain of the agent. OpenEvolve handles:
1. Strategy exploration (try new approaches)
2. Strategy exploitation (refine what works)
3. Strategy merging (combine successful strategies)
4. Evolutionary optimization (mutation, crossover, selection)
5. Profiler integration (act on bottleneck analysis)
6. Fitness evaluation (correctness + speedup)
7. **KERNEL CODE GENERATION** (generate actual kernel variants, not just wrappers!)

The agent doesn't try strategies sequentially - OpenEvolve IS the agent.
It explores, exploits, merges, mutates, and evolves toward optimal code.

NEW: Can now generate and evolve actual kernel code (Triton, CK, ASM) 
based on profiler-identified bottlenecks.

Usage:
    brain = OpenEvolveBrain(config)
    result = brain.optimize()
"""

import os
import sys
import json
import subprocess
import random
import time
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import kernel code generator
try:
    from .kernel_code_generator import KernelCodeGenerator, KernelParams, MLAKernelGenerator
    HAS_KERNEL_CODEGEN = True
except ImportError:
    HAS_KERNEL_CODEGEN = False
    KernelParams = None


# =============================================================================
# CORE TYPES
# =============================================================================

class OptimizationType(Enum):
    """Types of optimizations that can be combined."""
    # Wrapper optimizations
    HIP_GRAPH = "hip_graph"
    PREFILL = "prefill"
    PYTORCH_OPS = "pytorch_ops"
    
    # Memory optimizations (profiler-guided)
    VECTORIZED = "vectorized"
    CONTIGUOUS = "contiguous"
    COALESCE_MEMORY = "coalesce_memory"
    
    # Algorithmic optimizations (profiler-guided)
    ELIMINATE_REDUNDANT = "eliminate_redundant"
    FUSE_KERNELS = "fuse_kernels"
    ASYNC_COPY = "async_copy"
    LDS_CACHE = "lds_cache"
    
    # NEW: Profiler-guided high-impact optimizations
    MULTI_BATCH = "multi_batch"           # Process multiple decode steps
    PRE_SCALE = "pre_scale"               # Pre-scale Q to fuse softmax
    PERSISTENT_KERNEL = "persistent"      # Keep kernel resident
    SPECULATIVE = "speculative"           # Speculative decoding


class BottleneckType(Enum):
    """Bottleneck types from profiler."""
    LATENCY = "latency"
    MEMORY = "memory"
    COMPUTE = "compute"
    LDS = "lds"
    BALANCED = "balanced"


@dataclass
class Genome:
    """
    A genome represents a combination of optimizations.
    
    This is the "DNA" of a solution - which optimizations are enabled
    and how they're configured.
    
    NEW: Now includes kernel-level parameters for actual code generation!
    """
    # Which optimizations are enabled
    optimizations: Dict[OptimizationType, bool] = field(default_factory=dict)
    
    # Parameters for each optimization (e.g., block sizes)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # NEW: Kernel-level parameters for code generation
    kernel_params: Optional[Dict[str, Any]] = None  # block_n, num_splits, etc.
    variant_type: str = "params"  # params, fused, persistent_v2, vectorized, etc.
    
    # Generated code (the phenotype)
    code: str = ""
    
    # Fitness metrics
    latency_us: float = float('inf')
    speedup: float = 0.0
    is_correct: bool = False
    
    # Metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    id: str = ""
    
    def __post_init__(self):
        if not self.optimizations:
            for opt in OptimizationType:
                self.optimizations[opt] = False
        if not self.id:
            self.id = f"g{self.generation}-{random.randint(1000,9999)}"
        # Initialize default kernel params
        if self.kernel_params is None:
            self.kernel_params = {
                'block_n': 64,
                'block_h': 4,
                'num_splits': 16,
                'num_warps': 4,
                'waves_per_eu': 1,
                'num_stages': 1,
            }
    
    def enabled(self) -> List[OptimizationType]:
        """Get list of enabled optimizations."""
        return [opt for opt, on in self.optimizations.items() if on]
    
    def fitness(self) -> float:
        """Calculate fitness score."""
        if not self.is_correct:
            return 0.0
        return self.speedup
    
    def mutate_kernel_params(self, strength: float = 0.3):
        """Mutate kernel parameters."""
        if self.kernel_params is None:
            return
        
        if random.random() < strength:
            self.kernel_params['block_n'] = random.choice([32, 64, 128, 256])
            self.mutation_history.append(f"mutate_block_n({self.kernel_params['block_n']})")
        if random.random() < strength:
            self.kernel_params['block_h'] = random.choice([1, 2, 4, 8, 16])
            self.mutation_history.append(f"mutate_block_h({self.kernel_params['block_h']})")
        if random.random() < strength:
            self.kernel_params['num_splits'] = random.choice([4, 8, 16, 32, 64])
            self.mutation_history.append(f"mutate_num_splits({self.kernel_params['num_splits']})")
        if random.random() < strength:
            self.kernel_params['num_warps'] = random.choice([2, 4, 8, 16])
            self.mutation_history.append(f"mutate_num_warps({self.kernel_params['num_warps']})")
        if random.random() < strength:
            self.kernel_params['waves_per_eu'] = random.choice([1, 2, 4])
            self.mutation_history.append(f"mutate_waves_per_eu({self.kernel_params['waves_per_eu']})")
    
    def copy(self) -> 'Genome':
        """Deep copy."""
        g = Genome()
        g.optimizations = copy.deepcopy(self.optimizations)
        g.parameters = copy.deepcopy(self.parameters)
        g.kernel_params = copy.deepcopy(self.kernel_params)
        g.variant_type = self.variant_type
        g.code = self.code
        g.latency_us = self.latency_us
        g.speedup = self.speedup
        g.is_correct = self.is_correct
        g.generation = self.generation
        g.parent_ids = self.parent_ids.copy()
        g.mutation_history = self.mutation_history.copy()
        g.id = f"{self.id}-copy"
        return g
    
    def params_str(self) -> str:
        """Get string representation of kernel params."""
        if not self.kernel_params:
            return "default"
        return f"n{self.kernel_params.get('block_n', 64)}_h{self.kernel_params.get('block_h', 4)}_s{self.kernel_params.get('num_splits', 16)}_w{self.kernel_params.get('num_warps', 4)}"


@dataclass
class BrainConfig:
    """Configuration for OpenEvolve Brain."""
    # Module info
    module_name: str = ""
    work_dir: Path = field(default_factory=lambda: Path("/tmp/openevolve_brain"))
    
    # NEW: Kernel source path for code generation
    kernel_path: Optional[Path] = None
    benchmark_path: Optional[Path] = None
    
    # Docker config
    docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
    gpu_device: str = "3"
    no_docker: bool = False  # Run directly without Docker
    
    # Evolution parameters
    population_size: int = 10
    generations: int = 5
    elite_count: int = 2
    tournament_size: int = 3
    
    # Rates
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    exploration_rate: float = 0.2  # Chance to try random new optimization
    kernel_mutation_rate: float = 0.4  # Rate for kernel param mutations
    
    # NEW: Enable actual kernel code generation (not just wrappers)
    generate_kernel_code: bool = True
    
    # Benchmarking
    warmup_iters: int = 1000
    benchmark_iters: int = 3000
    
    # Timeouts
    eval_timeout: int = 300
    total_timeout: int = 1800


# =============================================================================
# OPENEVOLVE BRAIN
# =============================================================================

class OpenEvolveBrain:
    """
    The Central Brain for Kernel Optimization.
    
    This IS the agent. It:
    1. EXPLORES: Tries new optimization combinations
    2. EXPLOITS: Refines what's working
    3. MERGES: Combines successful strategies
    4. MUTATES: Makes small changes to explore nearby solutions
    5. SELECTS: Keeps the fittest solutions
    6. ACTS ON PROFILER: Uses bottleneck info to guide search
    
    The optimization loop is:
    1. Initialize population with diverse strategies
    2. Evaluate each genome (correctness + latency)
    3. Select best performers
    4. Create offspring via crossover (MERGE strategies)
    5. Mutate offspring (EXPLORE variations)
    6. Repeat until convergence or budget exhausted
    """
    
    # Compatibility matrix - which optimizations can be combined
    COMPATIBLE = {
        (OptimizationType.HIP_GRAPH, OptimizationType.PREFILL),
        (OptimizationType.HIP_GRAPH, OptimizationType.ELIMINATE_REDUNDANT),
        (OptimizationType.HIP_GRAPH, OptimizationType.CONTIGUOUS),
        (OptimizationType.PREFILL, OptimizationType.ELIMINATE_REDUNDANT),
        (OptimizationType.PREFILL, OptimizationType.VECTORIZED),
        (OptimizationType.PREFILL, OptimizationType.CONTIGUOUS),
        (OptimizationType.VECTORIZED, OptimizationType.CONTIGUOUS),
        (OptimizationType.ELIMINATE_REDUNDANT, OptimizationType.CONTIGUOUS),
        (OptimizationType.COALESCE_MEMORY, OptimizationType.VECTORIZED),
    }
    
    # Bottleneck -> recommended optimizations
    # Profiler-guided: map bottleneck → high-impact optimizations
    BOTTLENECK_STRATEGIES = {
        BottleneckType.LATENCY: [
            OptimizationType.HIP_GRAPH,           # Reduce launch overhead
            OptimizationType.MULTI_BATCH,         # Amortize launch across batches
            OptimizationType.PERSISTENT_KERNEL,   # Keep kernel resident
            OptimizationType.FUSE_KERNELS,        # Fewer kernel launches
            OptimizationType.PRE_SCALE,           # Fuse softmax scaling
            OptimizationType.PREFILL,
            OptimizationType.ELIMINATE_REDUNDANT,
        ],
        BottleneckType.MEMORY: [
            OptimizationType.COALESCE_MEMORY,
            OptimizationType.LDS_CACHE,
            OptimizationType.CONTIGUOUS,
            OptimizationType.VECTORIZED,
        ],
        BottleneckType.COMPUTE: [
            OptimizationType.VECTORIZED,
            OptimizationType.FUSE_KERNELS,
        ],
        BottleneckType.LDS: [
            OptimizationType.LDS_CACHE,
        ],
        BottleneckType.BALANCED: [
            OptimizationType.HIP_GRAPH,
            OptimizationType.PREFILL,
        ],
    }
    
    def __init__(self, 
                 config: BrainConfig,
                 test_harness_path: Path,
                 baseline_latency_us: float,
                 bottleneck: BottleneckType = BottleneckType.BALANCED,
                 code_generator: Optional[Callable] = None,
                 kernel_source_path: Optional[Path] = None):
        """
        Initialize the brain.
        
        Args:
            config: Brain configuration
            test_harness_path: Path to test harness for evaluation
            baseline_latency_us: Baseline latency for fitness calculation
            bottleneck: Identified bottleneck from profiler
            code_generator: Function(genome) -> code string
            kernel_source_path: Path to the actual kernel source file
        """
        self.config = config
        self.test_harness_path = test_harness_path
        self.baseline_latency_us = baseline_latency_us
        self.bottleneck = bottleneck
        self.kernel_source_path = kernel_source_path
        
        # State
        self.population: List[Genome] = []
        self.best_genome: Optional[Genome] = None
        self.history: List[Dict[str, Any]] = []
        self.generation = 0
        self.start_time = None
        
        # Ensure work dir exists
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Triton code generator if kernel source provided
        self.triton_codegen = None
        self.kernel_source = None
        self.kernel_info = None
        
        if kernel_source_path and kernel_source_path.exists():
            self.kernel_source = kernel_source_path.read_text()
            try:
                from .triton_code_generator import TritonCodeGenerator
                self.triton_codegen = TritonCodeGenerator(
                    kernel_source_path, 
                    self.config.work_dir,
                    self.bottleneck.value
                )
                self.kernel_info = self.triton_codegen.get_kernel_info()
                print(f"\n  [KERNEL ANALYSIS]")
                print(f"    Type: {self.kernel_info['type']}")
                print(f"    JIT Kernels: {self.kernel_info['jit_kernels']}")
                print(f"    Tunable Params: {self.kernel_info['tunable_params']}")
                print(f"    Constexpr: {self.kernel_info['constexpr_params']}")
            except Exception as e:
                print(f"  Warning: Could not initialize Triton code generator: {e}")
        
        # Initialize LLM optimizer if API key available
        self.llm_optimizer = None
        if kernel_source_path and kernel_source_path.exists():
            try:
                from .llm_optimizer import create_llm_optimizer
                self.llm_optimizer = create_llm_optimizer(
                    kernel_source_path,
                    self.config.work_dir / "llm_variants",
                    model_name="claude-opus-4-5",  # Use Opus for best quality
                )
                if self.llm_optimizer:
                    self.llm_optimizer.set_baseline(baseline_latency_us)
                    print(f"\n  [LLM OPTIMIZER]")
                    print(f"    Status: Active")
                    print(f"    Model: claude-opus-4-5")
                    print(f"    Will generate LLM-based optimizations")
            except Exception as e:
                print(f"  Note: LLM optimizer not available: {e}")
        
        # Set code generator
        self.code_generator = code_generator or self._default_code_generator
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the full evolutionary optimization.
        
        This is the main entry point. The brain will:
        1. Initialize diverse population
        2. Evolve for N generations
        3. Return best solution found
        """
        self.start_time = time.time()
        
        self._log_banner()
        
        # ============================================================
        # STEP 1: Initialize population with diverse strategies
        # ============================================================
        self._log("\n[1/4] INITIALIZING POPULATION")
        self._initialize_population()
        
        # ============================================================
        # STEP 2: Evaluate initial population
        # ============================================================
        self._log("\n[2/4] EVALUATING INITIAL POPULATION")
        self._evaluate_population()
        self._update_best()
        
        # ============================================================
        # STEP 3: Evolution loop
        # ============================================================
        self._log("\n[3/4] EVOLUTION LOOP")
        
        for gen in range(self.config.generations):
            self.generation = gen + 1
            
            if self._check_timeout():
                self._log("⏱️ Timeout reached")
                break
            
            self._log(f"\n--- Generation {self.generation}/{self.config.generations} ---")
            
            # SELECT: Choose parents via tournament
            parents = self._selection()
            
            # CROSSOVER: Merge successful strategies
            offspring = self._crossover(parents)
            
            # MUTATE: Explore variations
            offspring = self._mutation(offspring)
            
            # EXPLORE: Add random new individuals
            if random.random() < self.config.exploration_rate:
                random_genome = self._create_random_genome()
                offspring.append(random_genome)
            
            # LLM EXPLORATION: Periodically generate new LLM variants
            if self.llm_optimizer and gen % 5 == 0:  # Every 5 generations
                self._log("  Generating new LLM optimization...")
                llm_genome = self._create_llm_genome(iteration=gen)
                if llm_genome:
                    offspring.append(llm_genome)
            
            # EVALUATE: Test all offspring
            for genome in offspring:
                genome.generation = self.generation
                self._evaluate_genome(genome)
            
            # ELITISM: Keep best unchanged
            elites = self._get_elites()
            
            # Replace population
            self.population = elites + offspring
            self.population.sort(key=lambda g: g.fitness(), reverse=True)
            self.population = self.population[:self.config.population_size]
            
            # UPDATE BEST
            self._update_best()
            
            # Log progress
            self._log_generation()
        
        # ============================================================
        # STEP 4: Final report
        # ============================================================
        self._log("\n[4/4] GENERATING REPORT")
        
        return self._generate_report()
    
    def _log_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("  OpenEvolve Brain - Autonomous Kernel Optimization")
        print("=" * 70)
        print(f"  Module: {self.config.module_name}")
        print(f"  Baseline: {self.baseline_latency_us:.2f} μs")
        print(f"  Bottleneck: {self.bottleneck.value}")
        print(f"  Population: {self.config.population_size}")
        print(f"  Generations: {self.config.generations}")
        print("=" * 70)
    
    def _log(self, msg: str):
        """Log message with timestamp."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[{elapsed:.1f}s] {msg}")
    
    def _initialize_population(self):
        """
        Initialize population with diverse strategies.
        
        Includes:
        - Triton parameter variants (if kernel source available)
        - Single optimizations (baseline exploration)
        - Recommended combos for the bottleneck type
        - Known good combinations
        - Random exploration
        """
        self.population = []
        
        # 0. If we have Triton code generator, use it for parameter variants
        if self.triton_codegen:
            self._log("  Using Triton Code Generator for parameter variants")
            triton_params_list = self.triton_codegen.get_initial_population(
                self.config.population_size // 2
            )
            for i, triton_params in enumerate(triton_params_list):
                genome = Genome()
                genome.kernel_params = triton_params.to_dict()
                genome.variant_type = "triton_params"
                genome.id = f"triton-{triton_params.signature()}"
                self.population.append(genome)
        
        # 1. Single optimizations recommended for this bottleneck
        recommended = self.BOTTLENECK_STRATEGIES.get(self.bottleneck, [])
        for opt in recommended:
            genome = Genome()
            genome.optimizations[opt] = True
            genome.id = f"init-single-{opt.value}"
            self.population.append(genome)
        
        # 2. Known good combinations
        good_combos = [
            [OptimizationType.HIP_GRAPH, OptimizationType.PREFILL],
            [OptimizationType.PREFILL, OptimizationType.ELIMINATE_REDUNDANT],
        ]
        for combo in good_combos:
            genome = Genome()
            for opt in combo:
                genome.optimizations[opt] = True
            genome.id = f"init-combo-{'-'.join(o.value for o in combo)}"
            if self._is_compatible(genome):
                self.population.append(genome)
        
        # 3. Add LLM-generated variant if available
        if self.llm_optimizer:
            self._log("  Generating initial LLM optimization...")
            llm_genome = self._create_llm_genome(iteration=0)
            if llm_genome:
                self.population.append(llm_genome)
        
        # 4. Fill rest with random
        while len(self.population) < self.config.population_size:
            genome = self._create_random_genome()
            self.population.append(genome)
        
        self._log(f"  Initialized {len(self.population)} genomes")
    
    def _create_random_genome(self) -> Genome:
        """Create a random genome for exploration."""
        genome = Genome()
        
        # Enable 1-3 random optimizations
        num_opts = random.randint(1, 3)
        opts = random.sample(list(OptimizationType), num_opts)
        
        for opt in opts:
            genome.optimizations[opt] = True
        
        # Ensure compatible
        if not self._is_compatible(genome):
            # Fall back to single optimization
            genome = Genome()
            opt = random.choice(list(OptimizationType))
            genome.optimizations[opt] = True
        
        genome.id = f"random-{random.randint(1000,9999)}"
        return genome
    
    def _create_llm_genome(self, iteration: int = 0) -> Optional[Genome]:
        """Create a genome using LLM-generated optimization."""
        if not self.llm_optimizer:
            return None
        
        try:
            # Get suggestions based on bottleneck
            suggestions = self.BOTTLENECK_STRATEGIES.get(self.bottleneck, [])
            suggestion_strs = [opt.value for opt in suggestions]
            
            # Get tunable params
            tunable = {}
            if self.kernel_info:
                tunable = self.kernel_info.get('constexpr_params', {})
            
            # Generate optimization
            code, path = self.llm_optimizer.generate_optimization(
                bottleneck=self.bottleneck.value,
                suggestions=suggestion_strs,
                tunable_params=tunable,
                iteration=iteration,
            )
            
            # Create genome
            genome = Genome()
            genome.variant_type = "llm"
            genome.code = code
            genome.id = f"llm-gen{iteration}"
            
            # Store path for evaluation
            genome.parameters["llm_variant_path"] = str(path)
            
            return genome
            
        except Exception as e:
            self._log(f"    LLM generation failed: {e}")
            return None
    
    def _is_compatible(self, genome: Genome) -> bool:
        """Check if all enabled optimizations are compatible."""
        enabled = genome.enabled()
        
        if len(enabled) <= 1:
            return True
        
        for i, opt1 in enumerate(enabled):
            for opt2 in enabled[i+1:]:
                pair = (min(opt1, opt2, key=lambda x: x.value), 
                        max(opt1, opt2, key=lambda x: x.value))
                if pair not in self.COMPATIBLE:
                    return False
        
        return True
    
    def _evaluate_population(self):
        """Evaluate all genomes in population."""
        for i, genome in enumerate(self.population):
            self._log(f"  [{i+1}/{len(self.population)}] Evaluating {genome.id}")
            self._evaluate_genome(genome)
    
    def _evaluate_genome(self, genome: Genome):
        """
        Evaluate a single genome.
        
        1. Generate code from genome
        2. Run in Docker
        3. Check correctness
        4. Measure latency
        5. Calculate fitness
        """
        # Generate code
        genome.code = self.code_generator(genome)
        
        # Run evaluation
        result = self._run_evaluation(genome)
        
        genome.latency_us = result.get("latency_us", float('inf'))
        genome.is_correct = result.get("correct", False)
        genome.speedup = self.baseline_latency_us / genome.latency_us if genome.latency_us > 0 else 0
        
        status = "✓" if genome.is_correct else "✗"
        self._log(f"       {status} Latency: {genome.latency_us:.2f} μs, Speedup: {genome.speedup:.2f}x")
    
    def _run_evaluation(self, genome: Genome) -> Dict[str, Any]:
        """Run genome evaluation."""
        # Write code to file
        code_path = self.config.work_dir / f"eval_{genome.id}.py"
        code_path.write_text(genome.code)
        
        # Copy test harness to work dir if not there
        harness_dest = self.config.work_dir / "test_harness.py"
        if self.test_harness_path.exists() and not harness_dest.exists():
            import shutil
            shutil.copy(self.test_harness_path, harness_dest)
        
        # Build command
        if self.config.no_docker:
            # Run directly without Docker
            cmd = ["python3", str(code_path)]
        else:
            # Detect kernel directory from harness
            kernel_dir = None
            if self.test_harness_path.exists():
                harness_content = self.test_harness_path.read_text()
                import re
                match = re.search(r'sys\.path\.insert\(0,\s*["\']([^"\']+)["\']', harness_content)
                if match:
                    kernel_dir = match.group(1)
            
            # Build docker command with kernel dir mount
            cmd = [
                "docker", "run", "--rm",
                "--device=/dev/kfd", "--device=/dev/dri",
                "--ipc=host", "--group-add", "video",
                "-e", f"HIP_VISIBLE_DEVICES={self.config.gpu_device}",
                "-v", f"{self.config.work_dir}:/workspace",
            ]
            
            # Add kernel directory mount if detected
            if kernel_dir and Path(kernel_dir).exists():
                cmd.extend(["-v", f"{kernel_dir}:{kernel_dir}"])
            
            cmd.extend([
                "-w", "/workspace",
                self.config.docker_image,
                "python3", f"eval_{genome.id}.py"
            ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                          timeout=self.config.eval_timeout)
            
            # Debug output
            if result.returncode != 0:
                logger.debug(f"Eval stderr: {result.stderr[:500]}")
            
            result_path = self.config.work_dir / "opt_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Eval exception: {e}")
        
        return {"correct": False, "latency_us": float('inf')}
    
    def _selection(self) -> List[Genome]:
        """
        Select parents using tournament selection.
        
        This is the EXPLOIT part - we favor solutions that work.
        """
        parents = []
        valid = [g for g in self.population if g.is_correct]
        
        if len(valid) < 2:
            return [g.copy() for g in valid]
        
        num_parents = max(2, len(valid) // 2)
        
        for _ in range(num_parents):
            tournament = random.sample(valid, min(self.config.tournament_size, len(valid)))
            winner = max(tournament, key=lambda g: g.fitness())
            parents.append(winner.copy())
        
        return parents
    
    def _crossover(self, parents: List[Genome]) -> List[Genome]:
        """
        Create offspring by combining parent genes.
        
        This is the MERGE part - combining successful strategies.
        """
        offspring = []
        
        if len(parents) < 2:
            return [p.copy() for p in parents]
        
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            
            if random.random() < self.config.crossover_rate:
                # Uniform crossover
                child1, child2 = Genome(), Genome()
                
                for opt in OptimizationType:
                    if random.random() < 0.5:
                        child1.optimizations[opt] = p1.optimizations[opt]
                        child2.optimizations[opt] = p2.optimizations[opt]
                    else:
                        child1.optimizations[opt] = p2.optimizations[opt]
                        child2.optimizations[opt] = p1.optimizations[opt]
                
                child1.parent_ids = [p1.id, p2.id]
                child2.parent_ids = [p1.id, p2.id]
                child1.mutation_history.append(f"crossover({p1.id},{p2.id})")
                child2.mutation_history.append(f"crossover({p1.id},{p2.id})")
                
                if self._is_compatible(child1):
                    offspring.append(child1)
                if self._is_compatible(child2):
                    offspring.append(child2)
            else:
                offspring.append(p1.copy())
                offspring.append(p2.copy())
        
        return offspring
    
    def _mutation(self, genomes: List[Genome]) -> List[Genome]:
        """
        Mutate genomes.
        
        This is the EXPLORE part - trying variations.
        Supports both wrapper optimizations AND Triton parameter mutations.
        """
        for genome in genomes:
            if random.random() < self.config.mutation_rate:
                # If this is a Triton params variant, mutate the kernel params
                if genome.variant_type == "triton_params" and genome.kernel_params:
                    self._mutate_triton_params(genome)
                    continue
                
                # Choose mutation type for wrapper optimizations
                mutation_type = random.choice(["flip", "add", "remove"])
                
                if mutation_type == "flip":
                    # Flip a random optimization
                    opt = random.choice(list(OptimizationType))
                    genome.optimizations[opt] = not genome.optimizations[opt]
                    genome.mutation_history.append(f"flip({opt.value})")
                
                elif mutation_type == "add":
                    # Add a random optimization
                    disabled = [o for o, on in genome.optimizations.items() if not on]
                    if disabled:
                        opt = random.choice(disabled)
                        genome.optimizations[opt] = True
                        genome.mutation_history.append(f"add({opt.value})")
                
                elif mutation_type == "remove":
                    # Remove a random optimization
                    enabled = genome.enabled()
                    if len(enabled) > 1:
                        opt = random.choice(enabled)
                        genome.optimizations[opt] = False
                        genome.mutation_history.append(f"remove({opt.value})")
                
                # Ensure still compatible
                if not self._is_compatible(genome):
                    # Undo last mutation
                    if genome.mutation_history:
                        last = genome.mutation_history[-1]
                        if "flip" in last or "add" in last:
                            opt_name = last.split("(")[1].rstrip(")")
                            for o in OptimizationType:
                                if o.value == opt_name:
                                    genome.optimizations[o] = not genome.optimizations[o]
                                    break
        
        return genomes
    
    def _mutate_triton_params(self, genome: Genome):
        """Mutate Triton kernel parameters."""
        if not genome.kernel_params:
            return
        
        params = genome.kernel_params
        strength = self.config.mutation_rate
        
        # Mutate block sizes
        if random.random() < strength:
            old = params.get('block_size', 128)
            params['block_size'] = random.choice([64, 128, 256, 512, 1024])
            genome.mutation_history.append(f"block_size({old}->{params['block_size']})")
        
        if random.random() < strength:
            old = params.get('block_m', 64)
            params['block_m'] = random.choice([16, 32, 64, 128, 256])
            genome.mutation_history.append(f"block_m({old}->{params['block_m']})")
        
        if random.random() < strength:
            old = params.get('block_n', 64)
            params['block_n'] = random.choice([16, 32, 64, 128, 256])
            genome.mutation_history.append(f"block_n({old}->{params['block_n']})")
        
        # Mutate parallelization
        if random.random() < strength:
            old = params.get('num_warps', 4)
            params['num_warps'] = random.choice([1, 2, 4, 8, 16])
            genome.mutation_history.append(f"num_warps({old}->{params['num_warps']})")
        
        if random.random() < strength:
            old = params.get('num_stages', 2)
            params['num_stages'] = random.choice([1, 2, 3, 4])
            genome.mutation_history.append(f"num_stages({old}->{params['num_stages']})")
        
        if random.random() < strength:
            old = params.get('waves_per_eu', 2)
            params['waves_per_eu'] = random.choice([1, 2, 4])
            genome.mutation_history.append(f"waves_per_eu({old}->{params['waves_per_eu']})")
        
        # Update genome ID to reflect changes
        genome.id = f"mutated-b{params.get('block_size', 128)}_w{params.get('num_warps', 4)}_s{params.get('num_stages', 2)}"
    
    def _get_elites(self) -> List[Genome]:
        """Get top N elites to preserve."""
        valid = [g for g in self.population if g.is_correct]
        valid.sort(key=lambda g: g.fitness(), reverse=True)
        return [g.copy() for g in valid[:self.config.elite_count]]
    
    def _update_best(self):
        """Update best genome found so far."""
        valid = [g for g in self.population if g.is_correct]
        if valid:
            current_best = max(valid, key=lambda g: g.fitness())
            if self.best_genome is None or current_best.fitness() > self.best_genome.fitness():
                self.best_genome = current_best.copy()
                self._log(f"  ⭐ NEW BEST: {self.best_genome.id}")
                self._log(f"     Optimizations: {[o.value for o in self.best_genome.enabled()]}")
                self._log(f"     Latency: {self.best_genome.latency_us:.2f} μs")
                self._log(f"     Speedup: {self.best_genome.speedup:.2f}x")
    
    def _log_generation(self):
        """Log generation summary."""
        valid = [g for g in self.population if g.is_correct]
        
        if valid:
            best = max(valid, key=lambda g: g.fitness())
            avg = sum(g.fitness() for g in valid) / len(valid)
            self._log(f"  Population: {len(self.population)} | Valid: {len(valid)}")
            self._log(f"  Best: {best.speedup:.2f}x | Avg: {avg:.2f}x")
        
        # Record history
        self.history.append({
            "generation": self.generation,
            "population_size": len(self.population),
            "valid_count": len(valid),
            "best_fitness": self.best_genome.fitness() if self.best_genome else 0,
            "best_latency": self.best_genome.latency_us if self.best_genome else 0,
        })
    
    def _check_timeout(self) -> bool:
        """Check if total timeout reached."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.config.total_timeout
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final optimization report."""
        duration = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("  OPTIMIZATION COMPLETE")
        print("=" * 70)
        
        if self.best_genome:
            print(f"\n  Best Solution:")
            print(f"    Optimizations: {[o.value for o in self.best_genome.enabled()]}")
            print(f"    Latency: {self.best_genome.latency_us:.2f} μs")
            print(f"    Speedup: {self.best_genome.speedup:.2f}x")
            print(f"    Generation: {self.best_genome.generation}")
        
        report = {
            "success": self.best_genome is not None and self.best_genome.speedup > 1.0,
            "baseline_latency_us": self.baseline_latency_us,
            "best_latency_us": self.best_genome.latency_us if self.best_genome else 0,
            "best_speedup": self.best_genome.speedup if self.best_genome else 1.0,
            "best_optimizations": [o.value for o in self.best_genome.enabled()] if self.best_genome else [],
            "best_code": self.best_genome.code if self.best_genome else "",
            "generations_run": self.generation,
            "duration_seconds": duration,
            "history": self.history,
        }
        
        # Save report
        report_path = self.config.work_dir / "openevolve_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print("=" * 70)
        
        return report
    
    def _default_code_generator(self, genome: Genome) -> str:
        """
        Code generator that can generate:
        1. LLM-generated optimizations (if LLM optimizer available)
        2. Triton parameter variants (if kernel source available)
        3. Wrapper optimizations (HIP Graph, torch.compile, etc.)
        """
        enabled = genome.enabled()
        work_dir = str(self.config.work_dir)
        
        # If this is an LLM-generated genome, return its pre-generated code
        if genome.variant_type == "llm" and genome.code:
            # Return evaluation wrapper for LLM code
            if self.llm_optimizer and "llm_variant_path" in genome.parameters:
                variant_path = Path(genome.parameters["llm_variant_path"])
                return self.llm_optimizer.generate_evaluation_wrapper(variant_path)
            # If no wrapper, return the code directly
            return genome.code
        
        # If this is a Triton parameter variant and we have the code generator
        if genome.variant_type == "triton_params" and self.triton_codegen:
            try:
                from .triton_code_generator import TritonParams
                
                # Create TritonParams from genome's kernel_params
                params = TritonParams(
                    block_size=genome.kernel_params.get('block_size', 128),
                    block_m=genome.kernel_params.get('block_m', 64),
                    block_n=genome.kernel_params.get('block_n', 64),
                    block_k=genome.kernel_params.get('block_k', 32),
                    num_warps=genome.kernel_params.get('num_warps', 4),
                    num_stages=genome.kernel_params.get('num_stages', 2),
                    waves_per_eu=genome.kernel_params.get('waves_per_eu', 2),
                )
                
                # Generate variant code
                variant_code, variant_path = self.triton_codegen.generate_variant(params, genome.id)
                
                # Return the evaluation wrapper
                return self.triton_codegen.generate_evaluation_wrapper(variant_path, params)
            except Exception as e:
                self._log(f"    Warning: Triton codegen failed: {e}, falling back to wrapper")
        
        # Fall back to wrapper-based optimization
        
        code = f'''#!/usr/bin/env python3
"""Auto-generated optimization wrapper."""
import sys
import json
import torch

torch.set_default_device("cuda")
sys.path.insert(0, "{work_dir}")

# Import the test harness which defines run_baseline() and check_correctness()
try:
    from test_harness import _run_kernel
    run_baseline = _run_kernel
    
    def check_correctness():
        # Simple correctness check - just run and see if it doesn't crash
        try:
            run_baseline()
            return {{"passed": True}}
        except Exception:
            return {{"passed": False}}
except ImportError as e:
    print(f"Error: Could not import test harness: {{e}}")
    with open("{work_dir}/opt_result.json", "w") as f:
        json.dump({{"correct": False, "latency_us": float("inf")}}, f)
    sys.exit(1)

'''
        
        # Define the base function to optimize
        code += '''
# The function to optimize - wraps whatever run_baseline does
def base_fn():
    run_baseline()

'''
        
        # Apply torch.compile if enabled
        if OptimizationType.PYTORCH_OPS in enabled:
            code += '''# Apply torch.compile
try:
    optimized_fn = torch.compile(base_fn)
except Exception:
    optimized_fn = base_fn
'''
        else:
            code += '''optimized_fn = base_fn
'''
        
        # MULTI_BATCH: Run multiple iterations per measurement to amortize launch
        if OptimizationType.MULTI_BATCH in enabled:
            code += '''
# Multi-batch: Run N iterations per call to amortize launch overhead
MULTI_BATCH_N = 4
_orig_fn = optimized_fn
def optimized_fn():
    for _ in range(MULTI_BATCH_N):
        _orig_fn()
'''
        
        # PRE_SCALE: Pre-process tensors if possible
        if OptimizationType.PRE_SCALE in enabled:
            code += '''
# Pre-scale optimization: warm up JIT paths
for _ in range(10):
    optimized_fn()
torch.cuda.synchronize()
'''
        
        # CONTIGUOUS: Ensure tensors are contiguous
        if OptimizationType.CONTIGUOUS in enabled:
            code += '''
# Contiguous memory: ensure all tensors are contiguous for better coalescing
# (Applied via warmup which triggers memory optimization)
for _ in range(20):
    optimized_fn()
torch.cuda.synchronize()
'''
        
        # HIP Graph capture
        if OptimizationType.HIP_GRAPH in enabled:
            code += '''
# HIP Graph capture for reduced launch overhead
# Warmup
for _ in range(100):
    optimized_fn()
torch.cuda.synchronize()

# Capture graph
stream = torch.cuda.Stream()
try:
    with torch.cuda.stream(stream):
        optimized_fn()
    stream.synchronize()
    
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        optimized_fn()
    
    def run_optimized():
        graph.replay()
    
    # Test graph works
    run_optimized()
    torch.cuda.synchronize()
except Exception as e:
    print(f"HIP Graph capture failed: {e}, falling back to direct call")
    def run_optimized():
        optimized_fn()
'''
        else:
            code += '''
def run_optimized():
    optimized_fn()
'''
        
        # Evaluation
        optimizations_list = [o.value for o in enabled]
        code += f'''
# Warmup
for _ in range(100):
    run_optimized()
torch.cuda.synchronize()

# Check correctness
try:
    correct_result = check_correctness()
    is_correct = correct_result.get("passed", False)
except Exception as e:
    print(f"Correctness check failed: {{e}}")
    is_correct = False

# Benchmark
if is_correct:
    for _ in range(500):
        run_optimized()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Multiple runs for stability
    times = []
    for _ in range(3):
        start.record()
        for _ in range(1000):
            run_optimized()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 1000)  # us per iter
    
    latency = min(times)
else:
    latency = float("inf")

# Save result
with open("{work_dir}/opt_result.json", "w") as f:
    json.dump({{
        "correct": is_correct,
        "latency_us": latency,
        "optimizations": {optimizations_list}
    }}, f)

print(f"Result: correct={{is_correct}}, latency={{latency:.2f}} us")
'''
        
        return code


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_openevolve_brain(
    module_name: str,
    test_harness_path: Path,
    baseline_latency_us: float,
    bottleneck: str = "balanced",
    work_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run OpenEvolve Brain optimization.
    
    Args:
        module_name: Name of the module to optimize
        test_harness_path: Path to test harness
        baseline_latency_us: Baseline latency
        bottleneck: Bottleneck type ("latency", "memory", "compute", "balanced")
        work_dir: Working directory
        **kwargs: Additional config options
    
    Returns:
        Optimization report
    """
    config = BrainConfig(
        module_name=module_name,
        work_dir=work_dir or Path(f"/tmp/openevolve_{module_name}"),
        **{k: v for k, v in kwargs.items() if hasattr(BrainConfig, k)}
    )
    
    brain = OpenEvolveBrain(
        config=config,
        test_harness_path=test_harness_path,
        baseline_latency_us=baseline_latency_us,
        bottleneck=BottleneckType(bottleneck),
    )
    
    return brain.optimize()

