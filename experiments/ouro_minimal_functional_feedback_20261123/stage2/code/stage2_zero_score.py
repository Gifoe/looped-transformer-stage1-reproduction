"""Run the unchanged causal scorer on pinned zero-shot requests."""
import stage2_score


if __name__ == "__main__":
    stage2_score.APPLE_BENCHMARKS = dict.fromkeys(stage2_score.APPLE_BENCHMARKS, 0)
    stage2_score.main()
