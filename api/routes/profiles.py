"""Profile endpoints — list, create, and update speaker profiles."""

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

bp = Blueprint("profiles", __name__)


def _get_profiles_path() -> str:
    return current_app.config["PROFILES_PATH"]


@bp.route("/profiles")
def list_profiles():
    """Return all profiles with embedding vectors stripped (too large for list view)."""
    from profiles import load_profiles

    data = load_profiles(_get_profiles_path())

    # Strip vectors — replace with counts for list view.
    # Copy each profile to avoid mutating the loaded data.
    stripped = []
    for profile in data.get("profiles", []):
        p = {**profile}
        p["embeddings"] = len(profile.get("embeddings", []))
        p.pop("centroid", None)
        p["voice_variants"] = len(profile.get("voice_variants", []))
        stripped.append(p)

    return jsonify({"profiles": stripped})


@bp.route("/profiles", methods=["POST"])
def create_new_profile():
    """Create a new speaker profile. Body: {name, role}."""
    from profiles import load_profiles, save_profiles, create_profile

    body = request.get_json()
    if body is None:
        return jsonify({"error": "Missing JSON body"}), 400

    name = body.get("name")
    role = body.get("role")
    if not name or not role:
        return jsonify({"error": "Both 'name' and 'role' are required"}), 400

    path = _get_profiles_path()
    profiles = load_profiles(path)
    profile_id = create_profile(profiles, name, role)
    save_profiles(profiles, path)

    return jsonify({"profile_id": profile_id}), 201


@bp.route("/profiles/<profile_id>", methods=["PUT"])
def update_profile(profile_id: str):
    """Update a profile's name and/or role. Body: {name?, role?}."""
    from profiles import load_profiles, save_profiles

    body = request.get_json()
    if body is None:
        return jsonify({"error": "Missing JSON body"}), 400

    if "name" not in body and "role" not in body:
        return jsonify({"error": "Provide at least 'name' or 'role' to update"}), 400

    path = _get_profiles_path()
    profiles = load_profiles(path)

    # Find the profile
    target = None
    for p in profiles.get("profiles", []):
        if p["id"] == profile_id:
            target = p
            break

    if target is None:
        return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    if "name" in body:
        target["name"] = body["name"]
    if "role" in body:
        target["role"] = body["role"]
    target["updated"] = datetime.now(timezone.utc).isoformat()

    save_profiles(profiles, path)

    return jsonify({"success": True})
