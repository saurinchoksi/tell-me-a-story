"""Flask application factory for the tell-me-a-story API.

Serves session data, speaker profiles, and audio files.
The frontend (Vite dev server) proxies /api requests here.
"""

import sys
from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "sessions"
DEFAULT_PROFILES_PATH = str(PROJECT_ROOT / "data" / "speaker_profiles.json")

# Add src/ to path so identify.py and profiles.py can be imported by bare name
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # api package importable from any cwd


def create_app(sessions_dir=None, profiles_path=None, detectors=None) -> Flask:
    """Create and configure the Flask app.

    Args:
        sessions_dir: Path to sessions directory. Defaults to PROJECT_ROOT/sessions.
            Injectable for test isolation with temp directories.
        profiles_path: Path to speaker_profiles.json. Defaults to PROJECT_ROOT/data/speaker_profiles.json.
            Injectable for test isolation.
        detectors: Failure-mode detector list for the detections routes.
            Defaults to the src/detectors registry (resolved lazily). Injectable
            so tests don't need the gitignored data/name_roster.json.
    """
    app = Flask(__name__)

    app.config["SESSIONS_DIR"] = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
    app.config["PROFILES_PATH"] = profiles_path or DEFAULT_PROFILES_PATH
    app.config["DETECTORS"] = detectors  # None -> registry, resolved in the route

    # CORS — needed when serving production builds (dev uses Vite proxy)
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        pass  # flask-cors is optional; dev works fine without it via Vite proxy

    # Register blueprints under /api prefix
    from api.routes.sessions import bp as sessions_bp
    from api.routes.profiles import bp as profiles_bp
    from api.routes.audio import bp as audio_bp
    from api.routes.speakers import bp as speakers_bp
    from api.routes.notes import bp as notes_bp
    from api.routes.axial_labels import bp as axial_labels_bp
    from api.routes.detections import bp as detections_bp
    from api.routes.name_corrections import bp as name_corrections_bp

    app.register_blueprint(sessions_bp, url_prefix="/api")
    app.register_blueprint(profiles_bp, url_prefix="/api")
    app.register_blueprint(audio_bp, url_prefix="/api")
    app.register_blueprint(speakers_bp, url_prefix="/api")
    app.register_blueprint(notes_bp, url_prefix="/api")
    app.register_blueprint(axial_labels_bp, url_prefix="/api")
    app.register_blueprint(detections_bp, url_prefix="/api")
    app.register_blueprint(name_corrections_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"Sessions dir: {app.config['SESSIONS_DIR']}")
    print(f"Profiles path: {app.config['PROFILES_PATH']}")
    app.run(host="localhost", port=5002, debug=True)
