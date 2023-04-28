"""Generate Python wheels HTML page."""
import os
import logging
import argparse

import github3
import github3.session as session
import requests


def url_is_valid(url):
  """Check if a given URL is valid, i.e. it returns 200 OK when requested."""
  r = requests.get(url)
  if r.status_code != 200:
    print("Warning: HTTP code %s for url %s" % (r.status_code, url))
  return r.status_code == 200


def list_release_wheels(repo):
  """List latest release Python wheels."""
  gh = github3.GitHub(
      session=session
      .GitHubSession(default_connect_timeout=100, default_read_timeout=100)
  )
  repo = gh.repository(*repo.split("/"))
  wheels = []
  # Just list latest release wheels.
  release = repo.latest_release()
  for asset in release.assets():
    print(f"Validating {asset.name} with url: {asset.browser_download_url}")
    if asset.name.endswith(".whl") and url_is_valid(asset.browser_download_url):
      wheels.append(asset)
  return wheels


def create_wheels_html_page(wheels, site_path):
  """Update the wheel page"""
  new_html = ""
  for asset in wheels:
    new_html += '<a href="%s">%s</a><br>\n' % (
        asset.browser_download_url,
        asset.name,
    )
  os.makedirs(site_path, exist_ok=True)
  wheel_html_path = os.path.join(site_path, "wheels.html")
  print(f"Saving HTML wheels page: {wheel_html_path}")
  open(wheel_html_path, "w").write(new_html)


def main():
  logging.basicConfig(level=logging.WARNING)
  parser = argparse.ArgumentParser(
      description="Generate Python wheels HTML page, from latest JAX IPU github release."
  )
  parser.add_argument("--repo", type=str, default="graphcore-research/jax-experimental")
  parser.add_argument("--site-path", type=str, default="_site/")

  args = parser.parse_args()
  wheels = list_release_wheels(args.repo)
  create_wheels_html_page(wheels, args.site_path)


if __name__ == "__main__":
  main()
