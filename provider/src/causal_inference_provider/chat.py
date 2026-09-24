"""Inference-only M5PHET adapter for explicitly fitted, local study artifacts."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from .provider import CausalInferenceProvider, _digest


CHAT_PROMPT = "Report the configured ATE and its uncertainty."
UNCERTAINTY = "econml_statsmodels_HC1_normal"
REF_PATTERN = re.compile(r"causal-ate:([a-f0-9]{64})")
STUDY_SLOT = "study"


def state_directory():
    default = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "causal-inference-m5phet/studies"
    return Path(os.environ.get("CAUSAL_INFERENCE_STATE_DIR", default))


def _clock(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("An aware clock is required.")
    return parsed.astimezone(timezone.utc)


def _study_label(config):
    """The name a study already carries in its own identifying config, in the words a person would use to ask for it."""
    return f"{config['estimand']} of {config['treatment']} on {config['outcome']}"


def _study_aliases(body, label):
    """Ordinary phrasings for one retained study.

    Every phrasing names both roles, the study reference, or the artifact's own development marking. A phrase naming a
    single column is deliberately absent: it would let a question about another study ("the effect of rainfall on
    outcome") resolve to this one because one of its words happened to match."""
    config = body["config"]
    treatment, outcome = config["treatment"], config["outcome"]
    aliases = [body["state_ref"], body["digest"], body["digest"][:12],
               f"{treatment} on {outcome}",
               f"effect of {treatment} on {outcome}",
               f"average effect of {treatment} on {outcome}",
               f"effect of {treatment} upon {outcome}"]
    if body.get("development") is True:
        aliases += ["demo study", "synthetic study", "development study"]
    return [phrase for phrase in dict.fromkeys(aliases) if phrase != label]


def save_study(core, directory, *, development=False):
    """Persist only an already fitted result. Never performs or triggers fitting."""
    if core.state["phase"] != "FITTED":
        raise ValueError("Only a successful explicitly fitted study can be saved.")
    result = core.infer()
    diagnostics = result["payload"]["diagnostics"]
    body = {
        "schema": "causal-inference.study.v1",
        "task_id": "causal-ate.v1:" + diagnostics["config_sha256"],
        "available_at": datetime.now(timezone.utc).isoformat(),
        "development": bool(development),
        "config": core.state["config"],
        "population": {"data_sha256": diagnostics["data_sha256"], "n_rows": diagnostics["n_rows"]},
        "result": result,
    }
    raw = json.dumps(body, sort_keys=True, allow_nan=False).encode()
    digest = sha256(raw).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (digest + ".json")).open("xb") as output:
        output.write(raw)
    return "causal-ate:" + digest


class M5PHETCausalProvider:
    """Reports a fitted study, never estimates from text or during inference."""

    name = "causal_inference"

    def __init__(self, directory=None, *, state_refs=None):
        self.directory = Path(directory) if directory is not None else state_directory()
        if state_refs is None:
            configured = os.environ.get("CAUSAL_INFERENCE_STATE_REFS")
            state_refs = configured.split(",") if configured else [
                "causal-ate:" + path.stem for path in sorted(self.directory.glob("*.json"))
                if re.fullmatch(r"[a-f0-9]{64}", path.stem) and path.is_file() and not path.is_symlink()
            ]
        if not isinstance(state_refs, (list, tuple)) or any(
            not isinstance(ref, str) or REF_PATTERN.fullmatch(ref) is None for ref in state_refs
        ):
            raise ValueError("Administrator state allowlist must contain causal-ate digest references.")
        self._known_states = tuple(sorted(set(state_refs)))

    def capabilities(self):
        return {
            "provider": self.name,
            "backend": "EconML LinearDML / explicit offline fitted study",
            "operations": ["infer"], "families": ["causal_inference"],
            "output_kinds": ["causal_effect"], "uncertainty_methods": [UNCERTAINTY],
            "supported": [{"operation": "infer", "family": "causal_inference", "output_kind": "causal_effect"}],
            "fit_required": True, "known_states": list(self._known_states),
            "resource_limits": {"max_outputs": 1, "max_artifact_bytes": 1_000_000},
            "identification": "conditional_on_user_assumptions",
        }

    def load(self, state_ref):
        match = REF_PATTERN.fullmatch(state_ref) if isinstance(state_ref, str) else None
        if match is None:
            raise ValueError("Invalid causal study state reference.")
        if state_ref not in self._known_states:
            raise ValueError("Study is not in the administrator state allowlist.")
        path = self.directory / (match[1] + ".json")
        if path.is_symlink():
            raise ValueError("Study state files must not be symlinks.")
        with path.open("rb") as source:
            raw = source.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Study artifact exceeds the byte limit.")
        if sha256(raw).hexdigest() != match[1]:
            raise ValueError("Study artifact digest mismatch.")
        body = json.loads(raw)
        if not isinstance(body, dict) or body.get("schema") != "causal-inference.study.v1":
            raise ValueError("Unsupported study artifact schema.")
        core = CausalInferenceProvider()
        if core.load(body.get("config"))["status"] != "OK":
            raise ValueError("Artifact lacks explicit identifying config.")
        if (body.get("result", {}).get("status") != "OK"
                or body.get("task_id") != "causal-ate.v1:" + _digest(core.state["config"])):
            raise ValueError("Artifact is not a successful fitted study for this task.")
        _clock(body["available_at"])
        return body | {"state_ref": state_ref, "digest": match[1]}

    @staticmethod
    def _answer(status, reason=None, *, payload=None, population=None):
        return {"outputs": {"effect": {"status": status, "why": reason,
                                       "payload": deepcopy(payload), "uncertainty": UNCERTAINTY}},
                "population": deepcopy(population)}

    def infer(self, request, state):
        """Read-only result retrieval with task, assumption, population and clock binding."""
        if not isinstance(request, dict) or not isinstance(state, dict):
            return self._answer("INVALID_INPUT", "Request and loaded state must be mappings.")
        combination = (request.get("operation"), request.get("family"), request.get("output_kind"))
        if combination != ("infer", "causal_inference", "causal_effect"):
            return self._answer("UNSUPPORTED_TASK", "Only causal-effect infer is available; use the explicit fit CLI.")
        body = {k: v for k, v in state.items() if k not in ("state_ref", "digest")}
        try:
            intact = _digest(body) == state.get("digest")
        except (ValueError, TypeError):
            intact = False
        ref = request.get("fitted_state_ref")
        if (not intact or ref != state.get("state_ref")
                or ref != "causal-ate:" + str(state.get("digest"))
                or ref not in self.capabilities()["known_states"]):
            return self._answer("INVALID_INPUT", "Fitted-state identity mismatch.")
        core = CausalInferenceProvider()
        checked = core.load(request.get("parameters"))
        if checked["status"] != "OK":
            return self._answer(checked["status"], checked["reason"], population=state["population"])
        if (request.get("schema_version") != "m5phet.task.draft2"
                or request.get("provider_ref") != self.name
                or request.get("task_id") != state["task_id"]
                or core.state["config"] != state["config"]
                or request.get("output_schema") != {"targets": ["effect"]}
                or request.get("population") != state["population"]
                or request.get("state") != state["population"]):
            return self._answer("INVALID_INPUT", "Request differs from the fitted study; explicitly fit a new study.",
                                population=state["population"])
        try:
            available = _clock(state["available_at"])
            as_of = _clock(request["as_of"])
        except (KeyError, TypeError, AttributeError, ValueError):
            return self._answer("INVALID_INPUT", "Valid aware availability and request clocks are required.")
        if as_of < available:
            return self._answer("INPUT_UNAVAILABLE", "Study was not available at the requested clock.",
                                population=state["population"])
        return self._answer("OK", payload=state["result"]["payload"], population=state["population"])

    def _retained_studies(self):
        """Every study this provider can serve right now, keyed by label and read back from its own artifact.

        A study whose artifact no longer loads is not offered: the vocabulary is what is retained, not what was fitted
        once."""
        loaded = []
        for ref in self.capabilities()["known_states"]:
            try:
                loaded.append(self.load(ref))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        labels = [_study_label(body["config"]) for body in loaded]
        studies = {}
        for label, body in zip(labels, loaded):
            # Two studies can share estimand and roles and differ in adjustments or level; both must stay nameable.
            key = label if labels.count(label) == 1 else f"{label} ({body['digest'][:12]})"
            studies[key] = body
        return studies

    def chat_slots(self):
        """Declare the one thing a person chooses in ordinary words: which retained study to report.

        The vocabulary is enumerated from the retained artifacts themselves, so a question is resolved against exactly
        the studies this provider holds and every other name is refused. Nothing else here is enumerable. The estimand,
        the treatment and outcome roles, the adjustment set, the assumptions, the confidence level and the population are
        carried by the chosen study, not chosen by the person: the engine admits one estimand (ATE) and this adapter
        never fits, so declaring them would either pose a question with a single answer or open a slot whose admissible
        values cannot be listed. With nothing retained there is no vocabulary and no slot at all."""
        studies = self._retained_studies()
        if not studies:
            return []
        return [{"name": STUDY_SLOT, "type": "string", "allowed": sorted(studies),
                 "aliases": {label: _study_aliases(body, label) for label, body in studies.items()}}]

    def _selected_study(self, parameters, config):
        """Resolve declared slot values to exactly one retained study, refusing any other value by name.

        A study nobody retains is never replaced by the study that happens to be here: that substitution would answer a
        question about another population with this one's number."""
        if not isinstance(parameters, dict):
            raise ValueError("Resolved parameters must be a mapping of declared slot values.")
        studies = self._retained_studies()
        if not studies:
            raise ValueError("No fitted study is retained; explicitly fit one with the CLI before asking for a report.")
        undeclared = sorted(str(name) for name in set(parameters) - {STUDY_SLOT})
        if undeclared:
            raise ValueError("Undeclared chat parameters: " + ", ".join(undeclared))
        label = parameters.get(STUDY_SLOT)
        if not isinstance(label, str) or label not in studies:
            raise ValueError(f"No retained study is named {label!r}; this provider holds {sorted(studies)}.")
        state = studies[label]
        if config.get("state") and config["state"] != state["state_ref"]:
            raise ValueError("The selected state reference is not the named study; say which one you mean.")
        explicit = config.get("parameters")
        if isinstance(explicit, dict) and explicit and explicit != state["config"]:
            raise ValueError("Explicit parameters differ from the named study's identifying config.")
        return state

    def chat_request(self, prompt, data, config, parameters=None):
        """Bounded report command, not natural-language causal identification.

        `parameters` carries the workbench's already-resolved slot values, which can only SELECT one retained study: the
        identification, the population and the availability clock still come from that study's artifact, and no word of
        the question enters the request. Callers that resolve nothing keep the explicit path unchanged -- the exact
        report command, an explicit identifying config and an explicit population reference."""
        if parameters is None:
            if not isinstance(prompt, str) or prompt.strip().casefold() != CHAT_PROMPT.casefold():
                raise ValueError("Unsupported prompt. Use: " + CHAT_PROMPT)
        elif not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A question is required; resolved parameters do not stand in for one.")
        if not isinstance(config, dict) or any(config.get(k) != v for k, v in (
            ("provider", self.name), ("family", "causal_inference"), ("output_kind", "causal_effect"),
        )):
            raise ValueError("Select causal_inference / causal_effect explicitly.")
        if parameters is None:
            if not isinstance(config.get("parameters"), dict):
                raise ValueError("Explicit identifying parameters are required.")
            if not isinstance(data, dict):
                raise ValueError("Chat needs a fitted population reference; raw datasets require the explicit fit CLI.")
            state = self.load(config.get("state"))
            population, identifying = data, config["parameters"]
        else:
            state = self._selected_study(parameters, config)
            population, identifying = state["population"], state["config"]
            if isinstance(data, dict) and data:
                if data != population:
                    raise ValueError("The attached population is not the named study's fitted population.")
            elif not (data is None or (isinstance(data, (str, list, tuple, dict)) and not data)):
                raise ValueError("A named study reports its own population; raw datasets require the explicit fit CLI.")
        request = {
            "schema_version": "m5phet.task.draft2", "task_id": state["task_id"],
            "operation": "infer", "family": "causal_inference", "output_kind": "causal_effect",
            "provider_ref": self.name, "fitted_state_ref": state["state_ref"],
            "as_of": config.get("as_of") or datetime.now(timezone.utc).isoformat(),
            "state": deepcopy(population), "population": deepcopy(population),
            "parameters": deepcopy(identifying),
            "input_schema": {"kind": "fitted_study_population_reference"},
            "output_schema": {"targets": ["effect"]},
            "execution_constraints": {"partial_results": False},
        }
        request["request_id"] = "causal-report:" + _digest(request)
        return request

    def chat_examples(self):
        """List only explicitly prepared synthetic development studies."""
        examples = []
        for ref in self.capabilities()["known_states"]:
            try:
                state = self.load(ref)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if state.get("development") is not True:
                continue
            # The example must resolve against this provider's OWN slots: an example a person clicks and that is then
            # refused teaches them the product is broken. It therefore names the study the way the slot declares it.
            label = next((name for name, body in self._retained_studies().items()
                          if body["digest"] == state["digest"]), None)
            prompt = f"Report {label}, with its uncertainty." if label else CHAT_PROMPT
            examples.append({
                "title": "SYNTHETIC/DEVELOPMENT: confounded ATE, known effect 2",
                "prompt": prompt, "data": deepcopy(state["population"]),
                "config": {"input": "json", "provider": self.name, "family": "causal_inference",
                           "output_kind": "causal_effect", "state": ref,
                           "as_of": datetime.now(timezone.utc).isoformat(),
                           "parameters": deepcopy(state["config"])},
            })
        return examples
