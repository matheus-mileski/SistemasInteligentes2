import pygame
from FlappyBird import *
import random
from MLP import MLP
import numpy as np
import os

# os.environ["SDL_VIDEODRIVER"] = "dummy"

BASE = 510
PIPE_INICIAL = 250
POPULACAO = 1
MAX_GERACOES = 100


def getInputs(bird, pipe, norm, input_indices):
    # All possible input values
    all_inputs = [
        bird.y / norm,
        (bird.y - pipe.height) / norm,
        (bird.y - pipe.top) / norm,
        (bird.y - pipe.bottom) / norm,
        (bird.x - pipe.x) / norm,
        bird.tick_count,
    ]

    # Selecting the desired inputs
    selected_inputs = [all_inputs[idx] for idx in input_indices]

    # Convert to numpy array and return
    return np.array(selected_inputs).reshape(-1, 1)


def main(params):
    print(params)
    adicionarCano = False
    geracao = 1
    birdBrains = []
    birds = []

    input_indices = params["input_indices"]
    norm = params["norm"]

    for i in range(POPULACAO):
        birdBrains.append(
            MLP(
                params["n_input"],
                params["n_hidden"],
                1,
                params["learn_rate"],
                # params["bias"],
            )
        )
        birds.append(Bird(50, random.randrange(300, 350)))

    removidos = []

    fundo = Fundo(FlappyBird.LARGURA)
    chao = Base(FlappyBird.LARGURA, BASE)
    canos = []
    numeroDeCanos = int(FlappyBird.LARGURA / 300) + 1
    for i in range(numeroDeCanos):
        canos.append(Pipe(PIPE_INICIAL + i * 300))
    cano_largura = canos[0].PIPE_TOP.get_width()
    score = 0
    max_score = 0
    window = pygame.display.set_mode((FlappyBird.LARGURA, FlappyBird.ALTURA))
    Clock = pygame.time.Clock()
    run = True
    while run:
        Clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if (
                len(canos) > 1
                and birds[0].x > canos[0].x + canos[0].PIPE_TOP.get_width()
            ):
                pipe_ind = 1
        elif geracao < MAX_GERACOES:
            birdBrains = removidos
            for i in range(POPULACAO):
                birds.append(Bird(50, 350))

            removidos = []

            canos = []
            for i in range(numeroDeCanos):
                canos.append(Pipe(PIPE_INICIAL + i * 300))
            score = 0
            geracao += 1
        else:
            run = False
            break

        # Preve os próximo movimento e executa o movimento
        for x, bird in enumerate(birds):
            inputs = getInputs(bird, canos[pipe_ind], norm, input_indices)
            # print(inputs)
            _, output = birdBrains[x].feedForward(inputs)
            # print(output[0][0])
            if output[0][0] > 0.5:
                bird.jump()
            bird.move()

        canosRemovidos = []
        # Faz a colisão e executa o aprendizado
        for cano in canos:
            for x, bird in enumerate(birds):
                collided, top = cano.collide(bird)
                if collided:
                    if top:
                        print("Morreu Cano Topo")
                        inputs = getInputs(bird, canos[pipe_ind], norm, input_indices)
                        print(inputs)
                        birdBrains[x].backPropagation(
                            inputs,
                            0,
                        )
                    else:
                        print("Morreu Cano Baixo")
                        inputs = getInputs(bird, canos[pipe_ind], norm, input_indices)
                        print(inputs)
                        birdBrains[x].backPropagation(
                            inputs,
                            1,
                        )
                    birds.pop(x)

                    removidos.append(birdBrains[x])
                    birdBrains.pop(x)

                if not cano.passed and (cano.x + cano_largura) < 0:
                    cano.passed = True
                    adicionarCano = True

            if (cano.x + cano_largura) < 0:
                canosRemovidos.append(cano)
            cano.move()

        if adicionarCano:
            score += 1
            if score > max_score:
                max_score = score
            canos.append(Pipe(numeroDeCanos * 300 - cano_largura))
            adicionarCano = False
        # Remove os canos fora da tela
        for cano in canosRemovidos:
            canos.remove(cano)

        # Checa Pássaro por pássaro se tocou o solo ou saiu da tela
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= BASE or bird.y < 0:
                birds.pop(x)
                removidos.append(birdBrains[x])

                if bird.y < 0:
                    print("Morreu Teto")
                    inputs = getInputs(bird, canos[pipe_ind], norm, input_indices)
                    print(inputs)
                    birdBrains[x].backPropagation(
                        inputs,
                        0,
                    )
                else:
                    print("Morreu Chão")
                    inputs = getInputs(bird, canos[pipe_ind], norm, input_indices)
                    print(inputs)
                    birdBrains[x].backPropagation(
                        inputs,
                        1,
                    )
                birdBrains.pop(x)

        fundo.move()
        chao.move()

        for b in birdBrains:
            b.setScore(score)

        draw_window(
            window,
            birds,
            canos,
            chao,
            fundo,
            score,
            geracao,
            len(birdBrains),
            max_score,
        )

    print("Acabou...")
    print(f"Max Score: {max_score}")
    for bird in removidos:
        if bird.getParametros()["score"] == max_score:
            print(bird.getParametros())

    return max_score


if __name__ == "__main__":
    params = {
        "n_input": 6,
        "input_indices": [0, 1, 2, 3, 4, 5],
        "n_hidden": 200,
        "learn_rate": 0.1,
        "norm": 10,
        "bias": 1,
    }
    main(params)
