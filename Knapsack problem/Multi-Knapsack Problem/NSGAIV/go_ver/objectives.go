// objectives.go
package nsga4

import "math"

// Objective set (exported):
//   - ObjLoadBalance: 標準差(負載率) → 越小越好
//   - ObjLatency:     最大負載率       → 越小越好
//   - ObjCost:        負載率平方和     → 越小越好
//
// 用法：
//   objectives := DefaultObjectives()
//   engine, _ := NewNSGA4(numServers, numRequests, population, generations, objectives, items, caps)


// ObjLoadBalance 最小化「負載率標準差」(讓分佈更均衡)
func ObjLoadBalance(solution []int, usage, items, caps [][]float64) float64 {
	r := flatten(div(usage, caps))
	return std(r)
}

// ObjLatency 最小化「最大負載率」(壓低尖峰)
func ObjLatency(solution []int, usage, items, caps [][]float64) float64 {
	r := flatten(div(usage, caps))
	mx := -math.MaxFloat64
	for _, v := range r {
		if v > mx {
			mx = v
		}
	}
	return mx
}

// ObjCost 最小化「負載率平方和」(凸懲罰，抑制局部過高負載)
func ObjCost(solution []int, usage, items, caps [][]float64) float64 {
	r := flatten(div(usage, caps))
	sum := 0.0
	for _, v := range r {
		sum += v * v
	}
	return sum
}

// DefaultObjectives 方便一次帶入三個目標
func DefaultObjectives() []ObjectiveFunc {
	return []ObjectiveFunc{ObjLoadBalance, ObjLatency, ObjCost}
}

/* ----------------------------- helpers ----------------------------- */

// div: element-wise usage/caps
func div(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			out[i][j] = a[i][j] / b[i][j]
		}
	}
	return out
}

// flatten 2D -> 1D
func flatten(m [][]float64) []float64 {
	if len(m) == 0 {
		return nil
	}
	out := make([]float64, 0, len(m)*len(m[0]))
	for i := range m {
		out = append(out, m[i]...)
	}
	return out
}

// std 標準差（母體）
func std(x []float64) float64 {
	n := float64(len(x))
	if n == 0 {
		return 0
	}
	sum, sum2 := 0.0, 0.0
	for _, v := range x {
		sum += v
		sum2 += v * v
	}
	mean := sum / n
	// Var = E[X^2] - (E[X])^2
	return math.Sqrt(sum2/n - mean*mean)
}