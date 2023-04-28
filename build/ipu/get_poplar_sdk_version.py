"""Get Poplar SDK version outputted."""
import re
import subprocess


def ipu_get_sdk_version() -> str:
  """Get SDK version from popc command line."""
  result = subprocess.run(['popc', '--version'], stdout=subprocess.PIPE)
  popc_output = result.stdout.decode("utf-8")
  sdk_version = re.search(r'POPLAR version\s*([\d.]+)', popc_output).group(1).strip()
  return sdk_version


def main():
  sdk_version = ipu_get_sdk_version()
  # Print short SDK version, stripped of .
  print(sdk_version.replace(".", "").strip())


if __name__ == "__main__":
  main()
