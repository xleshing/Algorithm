// nsga4.go
package nsga4

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

type ObjectiveFunc func(solution []int, usage [][]float64, itemValues [][]float64, capacities [][]float64) float64

type NSGA4 struct {
	NumServers     int
	NumRequests    int
	PopulationSize int
	Generations    int

	ItemValues       [][]float64 // shape: [numRequests][1]  (仍支援 >1 維)
	ServerCapacities [][]float64 // shape: [numServers][1]

	Objectives []ObjectiveFunc

	Population [][]int // 族群（每個解是長度 numRequests 的指派向量）
}

func NewNSGA4(
	numServers, numRequests, populationSize, generations int,
	objectives []ObjectiveFunc,
	itemValues [][]float64,
	serverCapacities [][]float64,
) (*NSGA4, error) {
	if numServers <= 0 || numRequests <= 0 || populationSize <= 0 || generations <= 0 {
		return nil, errors.New("invalid basic parameters")
	}
	if len(itemValues) != numRequests {
		return nil, errors.New("itemValues length must equal numRequests")
	}
	if len(serverCapacities) != numServers {
		return nil, errors.New("serverCapacities length must equal numServers")
	}
	rand.Seed(time.Now().UnixNano())

	ns := &NSGA4{
		NumServers:       numServers,
		NumRequests:      numRequests,
		PopulationSize:   populationSize,
		Generations:      generations,
		ItemValues:       itemValues,
		ServerCapacities: serverCapacities,
		Objectives:       objectives,
	}
	// 初始族群：保證可行
	pop := make([][]int, populationSize)
	for i := 0; i < populationSize; i++ {
		s, err := ns.generateFeasibleSolution()
		if err != nil {
			return nil, err
		}
		pop[i] = s
	}
	ns.Population = pop
	return ns, nil
}

/* ---------------------------- 基本工具 ---------------------------- */

func (ns *NSGA4) computeUsage(solution []int) [][]float64 {
	numObj := len(ns.ItemValues[0])
	usage := make2D(ns.NumServers, numObj)
	for j := 0; j < ns.NumServers; j++ {
		for i := 0; i < ns.NumRequests; i++ {
			if solution[i] == j {
				for d := 0; d < numObj; d++ {
					usage[j][d] += ns.ItemValues[i][d]
				}
			}
		}
	}
	return usage
}

func (ns *NSGA4) isFeasible(usage [][]float64) bool {
	for j := 0; j < ns.NumServers; j++ {
		for d := 0; d < len(usage[j]); d++ {
			if usage[j][d] > ns.ServerCapacities[j][d] {
				return false
			}
		}
	}
	return true
}

func (ns *NSGA4) generateFeasibleSolution() ([]int, error) {
	numObj := len(ns.ItemValues[0])
	usage := make2D(ns.NumServers, numObj)
	sol := make([]int, ns.NumRequests)

	perm := randPerm(ns.NumRequests)
	for _, i := range perm {
		feasible := make([]int, 0, ns.NumServers)
		for j := 0; j < ns.NumServers; j++ {
			ok := true
			for d := 0; d < numObj; d++ {
				if usage[j][d]+ns.ItemValues[i][d] > ns.ServerCapacities[j][d] {
					ok = false
					break
				}
			}
			if ok {
				feasible = append(feasible, j)
			}
		}
		if len(feasible) == 0 {
			// 這代表容量組合無法容納目前亂序；多嘗試幾次，最後仍失敗就回傳錯誤
			for retry := 0; retry < 20; retry++ {
				return ns.generateFeasibleSolution()
			}
			return nil, errors.New("no feasible solution under given capacities")
		}
		j := feasible[rand.Intn(len(feasible))]
		sol[i] = j
		for d := 0; d < numObj; d++ {
			usage[j][d] += ns.ItemValues[i][d]
		}
	}
	return sol, nil
}

func (ns *NSGA4) repairSolution(solution []int) []int {
	numObj := len(ns.ItemValues[0])
	usage := ns.computeUsage(solution)
	if ns.isFeasible(usage) {
		return solution
	}
	sol := append([]int(nil), solution...)
	improved := true

	for improved && !ns.isFeasible(usage) {
		improved = false
		for j := 0; j < ns.NumServers; j++ {
			over := false
			for d := 0; d < numObj; d++ {
				if usage[j][d] > ns.ServerCapacities[j][d] {
					over = true
					break
				}
			}
			if !over {
				continue
			}
			// 找在 j 上的 request
			indices := indicesWhere(sol, j)
			shuffle(indices)
			for _, req := range indices {
				// 試著搬到其他 server
				for k := 0; k < ns.NumServers; k++ {
					if k == j {
						continue
					}
					ok := true
					for d := 0; d < numObj; d++ {
						nj := usage[j][d] - ns.ItemValues[req][d]
						nk := usage[k][d] + ns.ItemValues[req][d]
						if nj > ns.ServerCapacities[j][d] || nk > ns.ServerCapacities[k][d] {
							ok = false
							break
						}
					}
					if ok {
						sol[req] = k
						for d := 0; d < numObj; d++ {
							usage[j][d] -= ns.ItemValues[req][d]
							usage[k][d] += ns.ItemValues[req][d]
						}
						improved = true
						break
					}
				}
				if ns.isFeasible(usage) {
					break
				}
			}
		}
	}
	if !ns.isFeasible(usage) {
		s, err := ns.generateFeasibleSolution()
		if err != nil {
			// 極端情況：回傳原解（仍可能違反），呼叫端可再處理
			return sol
		}
		return s
	}
	return sol
}

func (ns *NSGA4) fitness(solution []int) []float64 {
	u := ns.computeUsage(solution)
	out := make([]float64, len(ns.Objectives))
	for i, f := range ns.Objectives {
		out[i] = f(solution, u, ns.ItemValues, ns.ServerCapacities)
	}
	return out
}

/* ----------------------- 非支配排序 / 前沿 ------------------------ */

func (ns *NSGA4) fastNonDominatedSort(popFit [][]float64) [][]int {
	n := len(popFit)
	ranks := make([]int, n)
	domCount := make([]int, n)
	dominated := make([][]int, n)
	front := []int{}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if dominates(popFit[i], popFit[j]) {
				dominated[i] = append(dominated[i], j)
				domCount[j]++
			} else if dominates(popFit[j], popFit[i]) {
				dominated[j] = append(dominated[j], i)
				domCount[i]++
			}
		}
		if domCount[i] == 0 {
			ranks[i] = 0
			front = append(front, i)
		}
	}

	fronts := [][]int{front}
	i := 0
	for len(fronts[i]) > 0 {
		next := []int{}
		for _, p := range fronts[i] {
			for _, q := range dominated[p] {
				domCount[q]--
				if domCount[q] == 0 {
					ranks[q] = i + 1
					next = append(next, q)
				}
			}
		}
		i++
		fronts = append(fronts, next)
	}
	// 移除最後一個空 front
	if len(fronts) > 0 && len(fronts[len(fronts)-1]) == 0 {
		fronts = fronts[:len(fronts)-1]
	}
	return fronts
}

/* ------------------------ 距離：用負載率 ------------------------- */

func (ns *NSGA4) distByLoadRatio(sol1, sol2 []int) float64 {
	u1 := ns.computeUsage(sol1)
	u2 := ns.computeUsage(sol2)
	sum := 0.0
	for j := 0; j < ns.NumServers; j++ {
		for d := 0; d < len(u1[j]); d++ {
			r1 := u1[j][d] / ns.ServerCapacities[j][d]
			r2 := u2[j][d] / ns.ServerCapacities[j][d]
			diff := r1 - r2
			sum += diff * diff
		}
	}
	return math.Sqrt(sum)
}

/* ---------------------------- Selection --------------------------- */

func (ns *NSGA4) selection(parents, children [][]int) [][]int {
	pop := append(append([][]int{}, parents...), children...)
	popFit := make([][]float64, len(pop))
	for i := range pop {
		popFit[i] = ns.fitness(pop[i])
	}
	fronts := ns.fastNonDominatedSort(popFit)

	// Q1: 填滿 ≤ 0.5N
	type pair struct {
		sol     []int
		subarea string // "Q1"/"Q2"
	}
	Q := []pair{}
	total := 0
	r := 0
	for r < len(fronts) && total+len(fronts[r]) <= int(0.5*float64(ns.PopulationSize)) {
		for _, idx := range fronts[r] {
			Q = append(Q, pair{sol: pop[idx], subarea: "Q1"})
		}
		total += len(fronts[r])
		r++
	}

	// Q2: 從下一層補到 ≤ 1.5N（若不夠就盡量）
	limit := int(1.5 * float64(ns.PopulationSize))
	if r < len(fronts) {
		for _, idx := range fronts[r] {
			if total < limit {
				Q = append(Q, pair{sol: pop[idx], subarea: "Q2"})
				total++
			} else {
				break
			}
		}
	}

	L := len(Q)
	if L <= ns.PopulationSize {
		// 不需要刪除，直接回傳（保險起見截斷）
		out := make([][]int, 0, ns.PopulationSize)
		for i := 0; i < L && len(out) < ns.PopulationSize; i++ {
			out = append(out, Q[i].sol)
		}
		return out
	}

	// 距離矩陣
	dist := make2D(L, L)
	for i := 0; i < L; i++ {
		for j := i + 1; j < L; j++ {
			d := ns.distByLoadRatio(Q[i].sol, Q[j].sol)
			dist[i][j] = d
			dist[j][i] = d
		}
	}

	removed := make(map[int]bool)
	for L-len(removed) > ns.PopulationSize {
		minD := math.MaxFloat64
		mi, mj := -1, -1
		for i := 0; i < L; i++ {
			if removed[i] {
				continue
			}
			for j := i + 1; j < L; j++ {
				if removed[j] {
					continue
				}
				if Q[i].subarea == "Q1" && Q[j].subarea == "Q1" {
					continue
				}
				if dist[i][j] < minD {
					minD = dist[i][j]
					mi, mj = i, j
				}
			}
		}
		if mi < 0 {
			break
		}
		// 刪除規則
		if Q[mi].subarea == "Q2" && Q[mj].subarea == "Q2" {
			di := nearestAlive(dist, removed, mi)
			dj := nearestAlive(dist, removed, mj)
			if di < dj {
				removed[mi] = true
			} else {
				removed[mj] = true
			}
		} else if Q[mi].subarea == "Q1" && Q[mj].subarea == "Q2" {
			removed[mj] = true
		} else if Q[mi].subarea == "Q2" && Q[mj].subarea == "Q1" {
			removed[mi] = true
		} else {
			removed[mj] = true
		}
	}

	out := make([][]int, 0, ns.PopulationSize)
	for i := 0; i < L && len(out) < ns.PopulationSize; i++ {
		if !removed[i] {
			out = append(out, Q[i].sol)
		}
	}
	return out
}

/* ----------------------- Genetic operators ------------------------ */

func (ns *NSGA4) crossover(p1, p2 []int) ([]int, []int) {
	if rand.Float64() < 0.9 {
		point := 1 + rand.Intn(ns.NumRequests-1)
		c1 := append([]int{}, p1[:point]...)
		c1 = append(c1, p2[point:]...)
		c2 := append([]int{}, p2[:point]...)
		c2 = append(c2, p1[point:]...)
		return ns.repairSolution(c1), ns.repairSolution(c2)
	}
	return ns.repairSolution(append([]int{}, p1...)), ns.repairSolution(append([]int{}, p2...))
}

func (ns *NSGA4) mutation(sol []int) []int {
	s := append([]int{}, sol...)
	if rand.Float64() < 0.2 {
		idx := rand.Intn(ns.NumRequests)
		s[idx] = rand.Intn(ns.NumServers)
	}
	return ns.repairSolution(s)
}

/* ----------------------------- Evolve ----------------------------- */

func (ns *NSGA4) Evolve() (finalPareto [][]int, paretoHistory [][][]int) {
	parents := make([][]int, len(ns.Population))
	copy(parents, ns.Population)

	paretoHistory = make([][][]int, 0, ns.Generations)

	for g := 0; g < ns.Generations; g++ {
		children := make([][]int, 0, ns.PopulationSize*2)
		for i := 0; i < ns.PopulationSize; i += 2 {
			p1 := parents[i]
			p2 := parents[(i+1)%ns.PopulationSize]
			c1, c2 := ns.crossover(p1, p2)
			children = append(children, ns.mutation(c1), ns.mutation(c2))
		}
		if len(children) > ns.PopulationSize {
			children = children[:ns.PopulationSize]
		}

		parents = ns.selection(parents, children)

		// 記錄當代前沿
		popFit := make([][]float64, len(parents))
		for i := range parents {
			popFit[i] = ns.fitness(parents[i])
		}
		fronts := ns.fastNonDominatedSort(popFit)
		first := make([][]int, 0, len(fronts[0]))
		for _, idx := range fronts[0] {
			first = append(first, parents[idx])
		}
		paretoHistory = append(paretoHistory, first)
	}

	// 最終前沿
	popFit := make([][]float64, len(parents))
	for i := range parents {
		popFit[i] = ns.fitness(parents[i])
	}
	fronts := ns.fastNonDominatedSort(popFit)
	final := make([][]int, 0, len(fronts[0]))
	for _, idx := range fronts[0] {
		final = append(final, parents[idx])
	}
	return final, paretoHistory
}

/* ----------------------------- Helpers ---------------------------- */

func make2D(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}

func randPerm(n int) []int {
	a := make([]int, n)
	for i := 0; i < n; i++ {
		a[i] = i
	}
	shuffle(a)
	return a
}

func shuffle(a []int) {
	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

func indicesWhere(sol []int, val int) []int {
	out := []int{}
	for i, v := range sol {
		if v == val {
			out = append(out, i)
		}
	}
	return out
}

func dominates(a, b []float64) bool {
	// 全 <= 且 至少一維 <
	allLE := true
	anyLT := false
	for i := 0; i < len(a); i++ {
		if a[i] > b[i] {
			allLE = false
			break
		}
		if a[i] < b[i] {
			anyLT = true
		}
	}
	return allLE && anyLT
}

func nearestAlive(dist [][]float64, removed map[int]bool, i int) float64 {
	minD := math.MaxFloat64
	for k := 0; k < len(dist); k++ {
		if k == i || removed[k] {
			continue
		}
		if dist[i][k] < minD {
			minD = dist[i][k]
		}
	}
	return minD
}