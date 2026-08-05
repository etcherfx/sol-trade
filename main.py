import os
import shutil
import subprocess
import sys

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog

from sol_trade.config import config
from sol_trade.log import log_general, silence_console_logging
from sol_trade.trading import start_trading
from sol_trade.wallet import find_balance

config()


def check_json_state() -> bool:
    return bool(config().keypair and config().secondary_mints)


splash = r"""
.▄▄ ·       ▄▄▌  ▄▄▄▄▄▄▄▄   ▄▄▄· ·▄▄▄▄  ▄▄▄ .
▐█ ▀. ▪     ██•  •██  ▀▄ █·▐█ ▀█ ██▪ ██ ▀▄.▀·
▄▀▀▀█▄ ▄█▀▄ ██▪   ▐█.▪▐▀▀▄ ▄█▀▀█ ▐█· ▐█▌▐▀▀▪▄
▐█▄▪▐█▐█▌.▐▌▐█▌▐▌ ▐█▌·▐█•█▌▐█ ▪▐▌██. ██ ▐█▄▄▌
 ▀▀▀▀  ▀█▄▀▪.▀▀▀  ▀▀▀ .▀  ▀ ▀  ▀ ▀▀▀▀▀•  ▀▀▀ 
"""


def center_text(text, width):
    lines = text.splitlines()
    centered_lines = [line.center(width) for line in lines]
    return "\n".join(centered_lines)


def get_layout():
    terminal_width, _ = shutil.get_terminal_size()

    centered_splash = center_text(splash, terminal_width)
    centered_welcome = "Welcome to SolTrade! Select an option to proceed:".center(
        terminal_width
    )

    dialog_body = HSplit(
        [
            Window(
                content=FormattedTextControl(
                    text=centered_splash, style="class:splash"
                ),
                height=len(splash.splitlines()),
            ),
            Window(height=1, char=" "),  # Spacing
            Window(
                content=FormattedTextControl(
                    text=centered_welcome, style="class:welcome"
                ),
                height=1,
            ),
            Window(height=1, char=" "),  # Spacing
            HSplit(
                [
                    Button(
                        text="Start Trading",
                        handler=start_trading_handler,
                    ),
                    Button(
                        text="Exit",
                        handler=lambda: app.exit(result=False),
                    ),
                ],
                padding=1,
            ),
        ]
    )

    return Layout(
        HSplit(
            [
                Window(height=Dimension(preferred=1, weight=1)),  # Top padding
                Dialog(
                    title="SolTrade",
                    body=dialog_body,
                    with_background=False,
                ),
                Window(height=Dimension(preferred=1, weight=1)),  # Bottom padding
            ]
        )
    )


def start_trading_handler():
    # Exit the fullscreen UI so the terminal returns to the normal buffer
    app.exit(result="start_trading")


style = Style.from_dict(
    {
        "dialog": "bg:default #ffffff",
        "dialog frame.label": "bg:default #ffffff",
        "dialog.body": "bg:default #ffffff",
        "dialog shadow": "bg:default",
        "button.focused": "bg:#ffffff #000000",
        "splash": "bg:default #ffffff",
        "welcome": "bg:default #ffffff",
    }
)

# Define key bindings
kb = KeyBindings()


@kb.add("up")
@kb.add("down")
def _(event):
    event.app.layout.focus_next()


layout = get_layout()
app = Application(layout=layout, full_screen=True, style=style, key_bindings=kb)

app.output.show_cursor = lambda: None
app.output.hide_cursor()
result = app.run()

if result == "start_trading":
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True, check=False)
    silence_console_logging()
    can_run = check_json_state()

    try:
        log_general.info(
            f"SolTrade has detected {find_balance(config().primary_mint)} {config().primary_mint_symbol} tokens available for trading."
        )
    except Exception as e:  # noqa: BLE001 - balance fetch failure; log and exit
        log_general.error(f"Error finding {config().primary_mint_symbol} balance: {e}")
        sys.exit()

    if can_run:
        log_general.debug("SolTrade has successfully imported the API requirements.")
        start_trading()
    else:
        sys.exit()
elif not result:
    sys.exit()
