from FlappyBirdGA import FlappyBirdGA

if __name__ == "__main__":
    ga = FlappyBirdGA()
    best_solution, best_fitness = ga.run()

    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")
