import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
from MLP import MLP


class FlappyBird_Human:
    def __init__(self,mlp):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 700))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 145
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 15
        self.gravity = 10
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-200, 200)
        self.mlp = mlp
        self.score = 0

    def updateWalls(self):
        self.wallx -= 4
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-200, 200)

        if self.wallx + self.wallUp.get_width() < self.bird[0]:
            self.score +=1

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird):
            self.dead = True
        if downRect.colliderect(self.bird):
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 10

    def jumpAction(self):
        # Obter as coordenadas do pássaro e dos obstáculos
        bird_x = 70  
        bird_y = self.birdY
        wall_x = self.wallx
        wall_top_y = 360 + self.gap - self.offset
        wall_bottom_y = 0 - self.gap - self.offset

        # Calcular as distâncias e alturas relativas
        horizontal_distance = wall_x - bird_x
        height_difference_top = bird_y - wall_top_y
        height_difference_bottom = wall_bottom_y - bird_y

        # Normalizar as entradas para o intervalo [0, 1]
        normalized_horizontal_distance = horizontal_distance / 400  
        normalized_height_difference_top = height_difference_top / 700  
        normalized_height_difference_bottom = height_difference_bottom / 700  

        # Entradas para a MLP
        inputs = [normalized_horizontal_distance, normalized_height_difference_top, normalized_height_difference_bottom]

        # Executar a rede neural para obter a saída
        outputs = self.mlp.feedForward(np.array(inputs).reshape(-1, 1))[1]  

        # Decidir se o pássaro deve pular ou não com base nas saídas da MLP
        if outputs[0][0] > 0.5:  
            self.jump = 17
            self.gravity = 10
            self.jumpSpeed = 15


    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 10
                    self.jumpSpeed = 15

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
            self.screen.blit(font.render(str(self.counter),
                                         -1,
                                         (255, 255, 255)),
                             (200, 50))
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
            
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            pygame.display.update()

if __name__ == "__main__":
    mlp = MLP(3,15,1, taxaDeAprendizado=0.1)
    
    ga = GeneticAlgorithm(3,1)
    best_mlp = ga.execute()
    
    game = FlappyBird_Human(mlp)
    game.run()