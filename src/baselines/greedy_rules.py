"""
Browser environment for Pokemon game
"""
import nest_asyncio
from src.envs.game_env import PokemonBrowserEnv
nest_asyncio.apply()


class GreedyRulesBaseline:
    def __init__(self, env) -> None:
        self.env = env

    def navigate_title_screen(self):
        ocr_data = self.env.ocr()
        if any('continue' in s for s in ocr_data):
            self.navigate_titlescreen_continue()
        else:
            self.navigate_titlescreen_newgame()

    def navigate_titlescreen_newgame(self):
        print("New game")
        self.env.send_action("a")
        self.env.send_action("a")
        self.navigate_pc()

    def navigate_titlescreen_continue(self):
        print("Continue")
        self.env.send_action("a")
        self.env.send_action("a")

    def navigate_pc(self):
        # PC
        # Random pokemon
        self.env.send_action("left")

        for _ in range(6):
            self.env.send_action("a")

        # enter game
        self.env.send_action("up")
        self.env.send_action("up")
        self.env.send_action("a")

    def autoplay(self):
        self.navigate_title_screen()
        ocr_every = 1
        i = 0

        while True:
            if i % ocr_every == 0:
                ocr_data = self.env.ocr()
                print(ocr_data)
                # Cover moves - 'It won't have any effect.'
                if any("effect" in s for s in ocr_data):
                    self.env.send_action("down")

            self.env.send_action("a")
            i += 1


def main():
    env = PokemonBrowserEnv(image_resolution=(640, 360))
    env.connect()
    agent = GreedyRulesBaseline(env)
    try:
        agent.autoplay()
    except KeyboardInterrupt:
        print("Terminating")
        env.close()


if __name__ == "__main__":
    main()
