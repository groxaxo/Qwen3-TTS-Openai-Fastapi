const $ = (id) => document.getElementById(id);
const form = $("editor");
const fields = form.elements;
const local = {voices: {presets: [], profiles: [], can_clone: false}, selection: null, busy: false, file: null};
let inputUrl = "";
let outputUrl = "";

function status(message, error = false) {
  $("status").textContent = message;
  $("status").dataset.error = String(error);
}

async function request(path, options) {
  const response = await fetch("/studio/api/" + path, options);
  if (!response.ok) {
    const failure = await response.json().catch(() => ({}));
    const detail = failure.detail;
    throw new Error(typeof detail === "string" ? detail :
      Array.isArray(detail) ? detail.map((item) => item.msg).join(". ") :
      detail?.message || "Request failed (" + response.status + ").");
  }
  return response;
}

function source(url, label) {
  $("source-panel").hidden = !url;
  $("source-label").textContent = label;
  $("source").removeAttribute("src");
  if (url) $("source").src = url;
  $("source").load();
}

function clearResult() {
  $("result").removeAttribute("src");
  $("result").load();
  $("result-panel").hidden = true;
  if (outputUrl) URL.revokeObjectURL(outputUrl);
  outputUrl = "";
}

function select(voice = null) {
  clearResult();
  if (inputUrl) URL.revokeObjectURL(inputUrl);
  inputUrl = "";
  local.file = null;
  local.selection = voice;
  fields.reference.value = "";
  fields.name.value = voice?.name || "";
  fields.language.value = voice?.language || "Auto";
  if (!fields.language.value) fields.language.value = "Auto";
  fields.transcript.value = voice?.transcript || "";
  fields.speaker_only.checked = voice?.speaker_only || false;
  fields.instruction.value = "";
  fields.name.readOnly = Boolean(voice?.preset);
  $("reference-settings").hidden = Boolean(voice?.preset);
  $("preset-settings").hidden = !voice?.preset;
  $("save").hidden = Boolean(voice?.preset);
  source(voice?.id && !voice.preset ? "/studio/api/reference/" + encodeURIComponent(voice.id) + "?v=" + encodeURIComponent(voice.revision) : "", "Saved reference");
  status(voice?.id && !voice.preset ? "Backend voice ID: clone:" + voice.name : "");
  drawLibrary();
}

function drawLibrary() {
  const voices = [
    ...local.voices.presets.map((preset) => ({name: preset, preset})),
    ...local.voices.profiles,
  ];
  $("voices").replaceChildren(...voices.map((voice) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "voice";
    button.disabled = local.busy;
    button.setAttribute("aria-current", String(voice.preset
      ? voice.preset === local.selection?.preset : voice.id === local.selection?.id));
    const icon = document.createElement("span");
    icon.className = "voice-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "▶";
    const label = document.createElement("span");
    label.textContent = voice.name;
    const detail = document.createElement("small");
    detail.textContent = voice.preset ? "Preset" : "Saved · " + voice.language;
    label.append(detail);
    button.append(icon, label);
    button.onclick = () => select(voice);
    return button;
  }));
}

function setBusy(value) {
  local.busy = value;
  for (const control of form.elements) control.disabled = value;
  $("new-voice").disabled = value;
  drawLibrary();
}

function payload() {
  const data = new FormData(form);
  data.delete("reference");
  if (local.file) data.set("reference", local.file);
  if (local.selection?.id) {
    data.set("profile_id", local.selection.id);
    data.set("revision", local.selection.revision);
  }
  if (local.selection?.preset) data.set("preset", local.selection.preset);
  return data;
}

async function useFile(file) {
  if (!file || local.busy) return;
  if (file.size > 100 * 1024 * 1024) {
    fields.reference.value = "";
    status("Choose a file up to 100 MiB.", true);
    return;
  }
  const body = new FormData();
  body.append("reference", file);
  setBusy(true);
  status("Preparing reference audio…");
  try {
    const response = await request("reference/normalize", {method: "POST", body});
    const trimmed = response.headers.get("X-Reference-Trimmed") === "true";
    local.file = new File([await response.blob()], "reference.wav", {type: "audio/wav"});
    if (inputUrl) URL.revokeObjectURL(inputUrl);
    inputUrl = URL.createObjectURL(local.file);
    clearResult();
    source(inputUrl, file.name + (trimmed ? " · First 30 seconds" : ""));
    status(trimmed
      ? "Reference trimmed to the first 30 seconds. Enter the transcript for that clip."
      : "Reference audio ready. Preview or save the voice.");
  } catch (error) {
    fields.reference.value = "";
    status(error.message, true);
  } finally {
    setBusy(false);
  }
}

$("new-voice").onclick = () => select();
fields.reference.onchange = () => useFile(fields.reference.files[0]);
$("dropzone").ondragover = (event) => {
  event.preventDefault();
  $("dropzone").classList.add("drag");
};
$("dropzone").ondragleave = () => $("dropzone").classList.remove("drag");
$("dropzone").ondrop = (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("drag");
  useFile(event.dataTransfer.files[0]);
};

form.onsubmit = async (event) => {
  event.preventDefault();
  if (local.busy) return;
  const body = payload();
  const supportsOpus = $("result").canPlayType('audio/ogg; codecs="opus"');
  body.set("response_format", supportsOpus ? "opus" : "mp3");
  clearResult();
  setBusy(true);
  status("Generating preview…");
  try {
    const response = await request("preview", {method: "POST", body});
    outputUrl = URL.createObjectURL(await response.blob());
    $("result").src = outputUrl;
    $("result-panel").hidden = false;
    status("Preview ready.");
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
};

$("save").onclick = async () => {
  const body = payload();
  setBusy(true);
  status("Saving voice…");
  try {
    const saved = await (await request("voices", {method: "POST", body})).json();
    local.voices = await (await request("voices")).json();
    select(local.voices.profiles.find((voice) => voice.id === saved.id));
    status("Saved. Use " + saved.voice + " in the speech API.");
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
};

request("voices").then((response) => response.json()).then((voices) => {
  local.voices = voices;
  $("connection").textContent = voices.can_clone ? "Local · Presets and reference voices" : "Local · Preset playback";
  $("connection").classList.add("ready");
  select(voices.presets.length ? {preset: voices.presets[0], name: voices.presets[0]} : null);
}).catch((error) => status(error.message, true));
