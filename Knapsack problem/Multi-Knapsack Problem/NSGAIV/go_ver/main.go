// main.go（示意）
package main

import (
	"fmt"
	"math/rand"
	"time"

	"nsga4"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	numServers := 12
	numRequests := 250
	population := 20
	generations := 100

	// 構造 itemValues / capacities（你會自己處理）
	items := make([][]float64, numRequests)
	for i := range items {
		items[i] = []float64{float64(1 + rand.Intn(4))}
	}
	caps := make([][]float64, numServers)
	for j := range caps {
		caps[j] = []float64{float64(50 + rand.Intn(11))}
	}

	objectives := nsga4.DefaultObjectives()

	engine, err := nsga4.NewNSGA4(numServers, numRequests, population, generations, objectives, items, caps)
	if err != nil {
		panic(err)
	}
	final, history := engine.Evolve()
	fmt.Println("Final Pareto size:", len(final))
	fmt.Println("History generations:", len(history))
}