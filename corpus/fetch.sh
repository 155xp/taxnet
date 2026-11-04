#!/usr/bin/env bash
# Download a held-out corpus: `fetch.sh` -> test, `fetch.sh test2` -> test2.
# test  : evaluated once the three networks were finished on dev.
# test2 : fetched afterwards, used once (the decorator rule was fixed after seeing two
#         Python errors in test, so test is no longer untouched).
# test3 : fetched after the fixes test2 prompted (BOM, CRLF comments, nested f-strings,
#         decorator expressions), used once for the final numbers.
# Each repo is pinned to the commit GitHub reports for its default branch at fetch time;
# the SHAs are written to corpus/<split>_repos.lock so the exact corpus can be re-fetched.
set -euo pipefail
SPLIT=${1:-test}
ROOT=${TAXNET_CORPUS:-$HOME/.cache/taxnet/corpus}
LOCK=$(dirname "$0")/${SPLIT}_repos.lock
mkdir -p "$ROOT/dl" "$ROOT/$SPLIT"
fetch() {  # lang owner/repo [sha]
  local lang=$1 repo=$2 sha=${3:-}
  if [ -z "$sha" ]; then
    sha=$(curl -fsSL "https://api.github.com/repos/$repo/commits/HEAD" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
  fi
  local name=${repo//\//__}
  local dest="$ROOT/$SPLIT/$lang/$name"
  if [ -d "$dest" ]; then echo "have $repo"; return; fi
  echo "fetch $lang $repo @ $sha"
  local tgz="$ROOT/dl/$name-$sha.tar.gz"
  curl -fsSL -o "$tgz" "https://codeload.github.com/$repo/tar.gz/$sha"
  mkdir -p "$dest.tmp"
  tar -xzf "$tgz" -C "$dest.tmp" --strip-components=1 --no-same-owner --no-same-permissions
  mv "$dest.tmp" "$dest"
  echo "$lang $repo $sha" >> "$LOCK"
}
if [ -f "$LOCK.pinned" ]; then
  while read -r lang repo sha; do fetch "$lang" "$repo" "$sha"; done < "$LOCK.pinned"
  exit 0
fi
if [ "$SPLIT" = test ]; then
for r in django/django sympy/sympy pandas-dev/pandas scikit-learn/scikit-learn; do fetch py $r; done
for r in prometheus/prometheus etcd-io/etcd gohugoio/hugo hashicorp/terraform golang/tools caddyserver/caddy cli/cli; do fetch go $r; done
for r in sqlite/sqlite redis/redis lua/lua curl/curl git/git python/cpython nginx/nginx vim/vim madler/zlib libuv/libuv jqlang/jq; do fetch c $r; done
elif [ "$SPLIT" = test2 ]; then
for r in pallets/flask psf/requests pytest-dev/pytest psf/black encode/httpx fastapi/fastapi scipy/scipy ansible/ansible; do fetch py $r; done
for r in gin-gonic/gin spf13/cobra moby/moby hashicorp/consul containerd/containerd junegunn/fzf; do fetch go $r; done
for r in openssl/openssl postgres/postgres tmux/tmux htop-dev/htop systemd/systemd libgit2/libgit2; do fetch c $r; done
else
for r in numpy/numpy pydantic/pydantic sqlalchemy/sqlalchemy pypa/pip sphinx-doc/sphinx scrapy/scrapy tornadoweb/tornado twisted/twisted; do fetch py $r; done
for r in go-gitea/gitea traefik/traefik minio/minio nats-io/nats-server syncthing/syncthing rclone/rclone; do fetch go $r; done
for r in FFmpeg/FFmpeg php/php-src ruby/ruby neovim/neovim raysan5/raylib bminor/glibc; do fetch c $r; done
fi
echo done
