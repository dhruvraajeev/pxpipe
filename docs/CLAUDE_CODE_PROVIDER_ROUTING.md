# Using other models with Claude Code

pxpipe supports **Claude Code only** to keep the maintenance burden small.
Routing Claude Code to other models is experimental. Kimi K3 on Cloudflare is
the only non-Anthropic model currently tested end to end.

Claude models use Anthropic by default. Two optional routes can run together:

- `OPENAI_MODELS` routes exact model IDs to OpenAI Responses.
- `CLOUDFLARE_MODELS` routes exact model IDs to Cloudflare's OpenAI-compatible endpoint.

If a model appears in both lists, precedence is:

```text
CLOUDFLARE_MODELS > OPENAI_MODELS > default routing
```

## Setup

```bash
OPENAI_UPSTREAM=https://api.openai.com \
OPENAI_API_KEY=your-openai-key \
OPENAI_MODELS=gpt-5.6-sol \
CLOUDFLARE_ACCOUNT_ID=your-account-id \
CLOUDFLARE_API_TOKEN=your-cloudflare-token \
CLOUDFLARE_MODELS=moonshotai/kimi-k3 \
npx pxpipe-proxy
```

This routes:

- `gpt-5.6-sol` to OpenAI Responses
- `moonshotai/kimi-k3` to Cloudflare
- every unlisted model ID to the default Anthropic route

The Cloudflare variables derive this OpenAI-compatible endpoint:

```text
https://api.cloudflare.com/client/v4/accounts/<account-id>/ai/v1
```

## Connect Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:47821 \
ANTHROPIC_AUTH_TOKEN=local-pxpipe \
claude --model claude-moonshotai/kimi-k3
```

pxpipe advertises Cloudflare model IDs with a `claude-` prefix because Claude
Code requires a Claude-shaped ID. The prefix is only an alias and is removed
before forwarding. Models can also be selected through Claude Code's `/model`
menu.

Verify discovery with:

```bash
curl http://127.0.0.1:47821/v1/models
```

Keep real provider credentials on the pxpipe process, not in Claude Code.
`PXPIPE_MODELS` is separate: it controls image compression, not routing.
