import asyncio
import base64
import os
import shlex
import io
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
from dotenv import load_dotenv
from mss import mss
from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

load_dotenv()

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]

class Resolution(TypedDict):
    width: int
    height: int

# sizes above XGA/WXGA are not recommended (see README.md)
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}

class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"

class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None

def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseAnthropicTool):
    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    def __init__(self):
        super().__init__()

        self.width = int(os.getenv("WIDTH") or 0)
        self.height = int(os.getenv("HEIGHT") or 0)
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
        else:
            self.display_num = None

        # Initialize caffeinate command base
        self.caffeinate = "caffeinate"
        # Initialize mss when the tool is created
        self.mss = mss()

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    async def screenshot(self) -> ToolResult:
        """Take a screenshot of the current screen using mss and return the base64 encoded image."""
        try:
            # Get the appropriate monitor
            monitor = self.mss.monitors[self.display_num + 1 if self.display_num is not None else 1]
            
            # Capture the screenshot using screencapture (part of macOS)
            screenshot_path = f"/tmp/screenshot_{uuid4()}.png"
            await self.shell(f"screencapture -x {screenshot_path}", take_screenshot=False)
            
            # Open and process the screenshot
            with Image.open(screenshot_path) as img:
                if self._scaling_enabled:
                    target_width, target_height = self.scale_coordinates(
                        ScalingSource.COMPUTER, self.width, self.height
                    )
                    if (target_width, target_height) != (self.width, self.height):
                        img = img.resize((target_width, target_height))
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
            
            # Clean up temporary file
            os.remove(screenshot_path)
            
            # Encode to base64
            base64_image = base64.b64encode(img_byte_arr).decode()
            
            return ToolResult(output="Screenshot taken successfully", error=None, base64_image=base64_image)
        except Exception as e:
            raise ToolError(f"Failed to take screenshot: {str(e)}")

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                return await self.shell(f"cliclick m:{x},{y}")
            elif action == "left_click_drag":
                return await self.shell(f"cliclick dd:{x},{y}")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Convert key notation to cliclick format
                key_mapping = {
                    "Return": "return",
                    "space": "space",
                    "Tab": "tab",
                    "Return": "⌤",  # Return key
                    "Enter": "⌤",   # Alternative name for Return
                    "space": " ",    # Space character
                    "Tab": "⇥",      # Tab key
                    "Backspace": "⌫", # Backspace key
                    "Delete": "⌦",    # Delete key
                    "Left": "←",      # Left arrow
                    "Right": "→",     # Right arrow
                    "Up": "↑",        # Up arrow
                    "Down": "↓",      # Down arrow
                    "Escape": "⎋",    # Escape key
                    "Home": "↖",      # Home key
                    "End": "↘",       # End key
                    "PageUp": "⇞",    # Page Up
                    "PageDown": "⇟",  # Page Down
                }
                mapped_key = key_mapping.get(text, text)
                return await self.shell(f"cliclick kd:{mapped_key}")
            elif action == "type":
                results: list[ToolResult] = []
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    cmd = f"cliclick t:{shlex.quote(chunk)}"
                    results.append(await self.shell(cmd, take_screenshot=False))
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(
                    output="".join(result.output or "" for result in results),
                    error="".join(result.error or "" for result in results),
                    base64_image=screenshot_base64,
                )

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                # Use cliclick to get cursor position
                result = await self.shell(
                    "cliclick p",
                    take_screenshot=False,
                )
                output = result.output or ""
                try:
                    x, y = map(int, output.strip().split(","))
                    x, y = self.scale_coordinates(
                        ScalingSource.COMPUTER, x, y
                    )
                    return result.replace(output=f"X={x},Y={y}")
                except ValueError:
                    raise ToolError("Failed to parse cursor position")
            else:
                click_command = {
                    "left_click": "c",
                    "right_click": "rc",
                    "middle_click": "mc",
                    "double_click": "dc",
                }[action]
                # Get current cursor position first
                pos_result = await self.shell("cliclick p", take_screenshot=False)
                if pos_result.output:
                    x, y = pos_result.output.strip().split(",")
                    return await self.shell(f"cliclick {click_command}:{x},{y}")
                else:
                    raise ToolError("Failed to get cursor position for click")

        raise ToolError(f"Invalid action: {action}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        # Wrap the command with caffeinate to prevent sleep
        caffeinated_command = f"{self.caffeinate} -i {command}"
        _, stdout, stderr = await run(caffeinated_command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)