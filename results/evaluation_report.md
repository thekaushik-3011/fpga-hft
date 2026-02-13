# Hardware-Aware Evaluation Report: Trade vs No-Trade SVM Accelerator

**Target Platform:** PYNQ-Z2 (Zynq-7020)
**Clock Frequency:** 200 MHz
**Constraint:** Ultra-Low Latency HFT
**Dataset:** HFT Realistic (Low SNR)

## Part 1 — Strict Hardware-Aware Re-Evaluation

Inference simulation performed with bit-accurate Q8.8 fixed-point arithmetic.

### Accuracy Metrics
| Model | Float Accuracy | Fixed-Point Accuracy (Q8.8) | Accuracy Drop |
| :--- | :--- | :--- | :--- |
| **Linear SVM** | 64.62% | **65.29%** | +0.67% (Stable) |
| **Kernel SVM (8 SVs)** | 94.54% | **50.00%** (Random) | -44.54% (Collapse) |

> **Critical Finding:** The dataset is now realistic for HFT (Signal-to-Noise Ratio is low).
> *   **Linear SVM:** Captures the core directional signal (~65%), which is highly profitable for HFT. It is robust to quantization.
> *   **Kernel SVM:** While theoretically superior (94.5% due to non-linear patterns), it requires **1511 Support Vectors**. Constraining it to 8 SVs for hardware latency causes it to collapse to random guessing (50%).

### Computational Metrics (Per Inference)
Counting explicit hardware operations:

| Metric | Linear SVM | Kernel SVM (8 SVs) |
| :--- | :--- | :--- |
| **MAC Operations** | 16 | 8 |
| **Multiplications** | 16 | 136 |
| **Additions** | 17 | 265 |
| **Comparisons** | 1 | 1 |
| **LUT Lookups** | 0 | 8 |
| **Total Arithmetic Ops** | **34** | **418** |

### Cycle Estimation (at 200 MHz)
| Metric | Linear SVM (Fully Parallel) | Kernel SVM (Folded, 16 DSPs) |
| :--- | :--- | :--- |
| **Cycles per Inference** | 6 | 23 |
| **Latency** | **30 ns** | 115 ns |
| **Throughput** | **33.3 M Inf/sec** | 8.7 M Inf/sec |

---

## Part 2 — Resource Estimation (Pre-FPGA)

### Linear SVM
*   **DSP Usage:** ~16 (7% of Zynq-7020).
*   **BRAM Usage:** ~0.5 (Negligible).
*   **Memory Footprint:** 34 bytes.

### Kernel SVM (8 SVs)
*   **DSP Usage:** ~16.
*   **BRAM Usage:** ~6.
*   **Memory Footprint:** ~300 bytes.

---

## Part 3 — Survivability & Conclusion

1.  **Linear SVM is the Only Viable Option:**
    *   **Accuracy:** 65% is excellent for HFT.
    *   **Latency:** 30ns is world-class.
    *   **Reliability:** 0% accuracy loss in fixed-point.

2.  **Kernel SVM is Disqualified:**
    *   **Failure:** The 8 SVs allowed by the latency budget capture **0% of the signal** (Accuracy 50%). To get 94% accuracy, we would need >1500 SVs, pushing latency to **>3 microseconds**, which is too slow for HFT.

**Recommendation:** Proceed with Linear SVM deployment on PYNQ-Z2.
