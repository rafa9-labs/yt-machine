"""
Wake-on-LAN — Send magic packets to wake a sleeping PC.

Usage:
  python -m tools.wake_pc                    # Uses WOL_MAC from .env
  python -m tools.wake_pc AA:BB:CC:DD:EE:FF  # Specific MAC address

Setup (target PC):
  1. Enter BIOS/UEFI → Enable "Wake on LAN" / "Power on by PCIe"
  2. Windows: Device Manager → Network Adapter → Power Management →
     Check "Allow this device to wake the computer"
  3. Some routers need a static ARP entry for the sleeping PC's IP/MAC
"""

import os
import sys
import socket
import struct
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

WOL_MAC = os.getenv("WOL_MAC", "")
WOL_BROADCAST = os.getenv("WOL_BROADCAST", "255.255.255.255")
WOL_PORT = int(os.getenv("WOL_PORT", "9"))


def send_wol(
    mac_address: str = None,
    broadcast: str = None,
    port: int = None,
) -> dict:
    """
    Send a Wake-on-LAN magic packet.

    The magic packet is a UDP broadcast containing:
    - 6 bytes of 0xFF (header)
    - 16 repetitions of the target MAC address (96 bytes)
    Total: 102 bytes

    Args:
        mac_address: Target MAC address (e.g., "AA:BB:CC:DD:EE:FF")
        broadcast: Broadcast address (default: 255.255.255.255)
        port: UDP port (default: 9, the standard WOL port)

    Returns:
        dict with success status
    """
    mac = mac_address or WOL_MAC
    bcast = broadcast or WOL_BROADCAST
    wol_port = port or WOL_PORT

    if not mac:
        return {"success": False, "error": "WOL_MAC not set. Add MAC address to .env"}

    # Clean MAC address (accept AA:BB:CC:DD:EE:FF or AA-BB-CC-DD-EE-FF or AABBCCDDEEFF)
    mac_clean = mac.replace(":", "").replace("-", "").replace(".", "").strip()
    if len(mac_clean) != 12:
        return {"success": False, "error": f"Invalid MAC address: {mac} (expected 12 hex chars)"}

    try:
        mac_bytes = bytes.fromhex(mac_clean)
    except ValueError:
        return {"success": False, "error": f"Invalid hex in MAC address: {mac}"}

    # Build magic packet: 6 × 0xFF + 16 × MAC
    magic_packet = b"\xff" * 6 + mac_bytes * 16

    print(f"  [WOL] Sending magic packet to {mac} via {bcast}:{wol_port}")
    print(f"  [WOL] Packet size: {len(magic_packet)} bytes")

    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Send 3 packets for reliability (some NICs need multiple)
        sent_count = 0
        for i in range(3):
            sock.sendto(magic_packet, (bcast, wol_port))
            sent_count += 1

        sock.close()

        print(f"  [WOL] ✅ Sent {sent_count} magic packets to {mac}")
        return {
            "success": True,
            "mac": mac,
            "packets_sent": sent_count,
            "broadcast": bcast,
            "port": wol_port,
        }

    except PermissionError:
        return {"success": False, "error": "Permission denied — try running as administrator"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mac = sys.argv[1] if len(sys.argv) > 1 else None
    result = send_wol(mac_address=mac)

    if result["success"]:
        print(f"✅ Wake-on-LAN sent to {result['mac']}")
    else:
        print(f"❌ Failed: {result['error']}")
        sys.exit(1)