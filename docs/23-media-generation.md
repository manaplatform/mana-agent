# Multimodal media generation

Mana-Agent supports optional image, speech/audio, and video generation through
the same model-driven gateway used by chat. Media generation is disabled by
default. Existing chat and repository workflows do not require media
configuration.

## Configure

Run `mana-agent --configure`. The Image generation, Voice generation, and Video
generation tabs each control the enabled state, provider,
capability-compatible model, credential reference, base URL, timeout, output
limit, and modality defaults. Credentials remain in Mana's existing
`~/.mana/secrets.toml`; media tables store only a credential key reference.

Example `~/.mana/config.toml`:

```toml
[media.image]
enabled = true
provider = "openai"
model = "gpt-image-1"
credential_ref = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
timeout_seconds = 120
max_output_bytes = 52428800

[media.image.defaults]
size = "1024x1024"
quality = "auto"
output_format = "png"

[media.voice]
enabled = true
provider = "openai"
model = "gpt-4o-mini-tts"
credential_ref = "OPENAI_API_KEY"
timeout_seconds = 120

[media.voice.defaults]
voice = "alloy"
output_format = "mp3"
speed = 1.0

[media.video]
enabled = true
provider = "openai"
model = "sora-2"
credential_ref = "OPENAI_API_KEY"
timeout_seconds = 600
max_output_bytes = 524288000
max_duration_seconds = 120

[media.video.defaults]
duration_seconds = 4
resolution = "720x1280"
```

Provider model metadata wins during capability normalization, followed by
Mana's maintained registry, then conservative non-text name inference. Unknown
models are not offered as supporting every modality. Selectors filter
independently for text, embedding, image, voice, and video use.

## Chat and durable jobs

Example requests:

```text
Create an image of a glass observatory above the clouds.
Generate a voice-over for this paragraph using the configured voice.
Create a four-second landscape video of waves at sunrise.
Check media generation media_abc123.
Cancel media generation media_abc123.
```

The entry model returns a typed media decision before execution. Invalid or
missing media decisions stop safely. Mana-Agent never substitutes a chat model,
another modality, or another provider.

Image and speech generation normally complete in the current turn. Video
creation returns a durable generation ID while the provider job is queued or
generating. A later status request polls the provider and downloads completed
content. Job records and non-image outputs remain beneath
`~/.mana/artifacts/media/`, so pending provider jobs can be inspected after
process restart. Completed image files alone are written
directly into the directory where Mana-Agent was launched, using safe
`media_*.png`, `media_*.jpg`, or `media_*.webp` names.

## Artifacts, events, and permissions

Media metadata, generation JSON, voice files, and video files remain in
`~/.mana/artifacts/media/`. Only completed image binaries are placed in the
launch directory; Mana-Agent does not create a workspace `.mana` directory for
media. Writes are atomic and enforce content MIME, maximum size, safe generated
names, SHA-256 identity, and path confinement.
The agent receives compact artifact metadata rather than base64 or binary
content. Copying an artifact into a repository remains a separate, explicit
repository write.

Media lifecycle events are `media_generation_requested`,
`media_generation_queued`, `media_generation_started`,
`media_generation_progress`, `media_generation_completed`,
`media_generation_failed`, and `media_generation_cancelled`. Events include
only safe job and artifact metadata; credentials, signed URLs, provider response
bodies, prompts, and binary data are excluded.

The exact scopes are `media.image.generate`, `media.voice.generate`,
`media.video.generate`, `media.artifact.write`, `media.status.read`, and
`media.generation.cancel`. A denied scope stops the exact operation. No
permission prompt is claimed unless a real request is created by a frontend
permission broker.

## Provider limits and troubleshooting

The built-in OpenAI adapter implements image generation, text-to-speech, video
job creation/status, and completed-output download. OpenAI's video deletion
endpoint applies to completed or failed jobs, so the adapter reports active-job
cancellation as unsupported instead of presenting deletion as cancellation. Custom
OpenAI-compatible endpoints may be used only when they implement the same
endpoint contract and declare a supported model capability. Reference-artifact
generation accepts one managed image artifact for the built-in image-edit and
video-reference endpoints; non-image, cross-session, missing, and multiple
references fail before a provider call. OpenAI video framing is selected through
the supported resolution values; a separate aspect-ratio parameter is rejected
rather than silently ignored.

Common errors are actionable:

- “Image generation is disabled.” Enable its section.
- “Media provider authentication is not configured.” Set the referenced secret.
- “The selected model does not support video generation.” Choose a compatible model.
- “The media generation timed out.” Increase the timeout or inspect provider health.
- “The generated output MIME type does not match its content.” No artifact was persisted.
