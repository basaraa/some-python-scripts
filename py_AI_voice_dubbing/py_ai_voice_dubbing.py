"""
Local AI Video Dubbing & Localization Studio (Entry Point)
--------------------------------------------------------
Modularized launcher for the offline AI dubbing & multi-voice narration studio.
"""

from modules.gui import create_app

if __name__ == "__main__":
    app = create_app()
    app.mainloop()
