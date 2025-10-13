"""
Browser environment for Pokemon game
"""
import nest_asyncio
nest_asyncio.apply()



def navigate_title_screen():
    ocr_data = env.ocr()
    
    if any('continue' in s for s in ocr_data):
        navigate_titlescreen_continue()
    else:
        navigate_titlescreen_newgame()

def navigate_titlescreen_newgame():
    print("New game")
    env.send_action("a")
    env.send_action("a")
    navigate_pc()

def navigate_titlescreen_continue():
    print("Continue")
    env.send_action("a")
    env.send_action("a")


def navigate_pc():
    # PC
    # Random pokemon
    env.send_action("left")

    for i in range(6):
        env.send_action("a")

    # enter game
    env.send_action("up")
    env.send_action("up")

    env.send_action("a")

def autoplay():
    navigate_title_screen()
    ocr_every = 1
    i = 0

    while True:
        if i % ocr_every == 0:
            ocr_data = env.ocr()
            print(ocr_data)
            # Cover moves - 'It won't have any effect.'
            if any("effect" in s for s in ocr_data):
                env.send_action("down")

        env.send_action("a")

        i += 1

    
def main():
    env = PokemonBrowserEnv(image_resolution=(640, 360))
    env.connect()
    try:
        autoplay()
    except KeyboardInterrupt:
        print("Terminating")
        env.close()


if __name__ == "__main__":
    main()
