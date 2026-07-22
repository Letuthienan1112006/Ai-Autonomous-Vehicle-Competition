"""Command-line entry point for the competition controller."""

from .competition_controller import main


if __name__ == "__main__":
    print("Starting lane following + traffic sign detection...")
    print("Press Q or ESC to exit")
    main()
