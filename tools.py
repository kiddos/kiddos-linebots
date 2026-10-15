import os
import logging
import requests

logger = logging.getLogger('uvicorn')


def control_ledstrip(brightness: int, mode: str) -> str:
    """Control the ESP32 LED strip brightness and mode.

    Args:
        brightness: Brightness level from 0 to 100.
        mode: Lighting mode ('white', 'rgb', or 'random').
    """
    base_url = os.getenv("LED_CONTROLLER_URL") or os.getenv("LEDSTRIP_URL")
    if not base_url:
        return "LED controller URL is not configured."
    base_url = base_url.rstrip("/")
    url = base_url if base_url.endswith("/api/ledstrip") else f"{base_url}/api/ledstrip"

    try:
        brightness_val = int(brightness)
        brightness_val = max(0, min(100, brightness_val))
    except (ValueError, TypeError):
        return f"Invalid brightness value '{brightness}'. Brightness must be an integer between 0 and 100."

    mode_str = str(mode).lower() if mode else ""
    if mode_str not in ["white", "rgb", "random"]:
        return f"Invalid mode '{mode}'. Mode must be 'white', 'rgb', or 'random'."

    payload = {
        "brightness": brightness_val,
        "mode": mode_str
    }
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return f"Successfully set LED strip brightness to {brightness_val} and mode to '{mode_str}'."
        else:
            return f"LED strip controller returned status code {res.status_code}: {res.text}"
    except Exception as e:
        logger.error(f"Error controlling LED strip: {e}")
        return f"Failed to control LED strip: {e}"


def read_room_dht11() -> str:
    """Read the room temperature (°C) and humidity (%) from the DHT11 sensor connected to the LED controller."""
    base_url = os.getenv("LED_CONTROLLER_URL") or os.getenv("LEDSTRIP_URL")
    if not base_url:
        return "LED controller URL is not configured."
    base_url = base_url.rstrip("/")
    url = base_url if base_url.endswith("/api/dht11") else f"{base_url}/api/dht11"

    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            temp = data.get("temperature")
            humidity = data.get("humidity")
            if temp is not None and humidity is not None:
                return f"Room temperature: {temp}°C, humidity: {humidity}%."
            else:
                return f"Unexpected data format received from sensor: {res.text}"
        else:
            return f"DHT11 sensor returned status code {res.status_code}: {res.text}"
    except Exception as e:
        logger.error(f"Error reading room DHT11 sensor: {e}")
        return f"Failed to read room temperature and humidity: {e}"
