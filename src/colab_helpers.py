from os import listdir, getcwd, chdir
from os.path import abspath, exists, join
from shutil import move
import subprocess

class ansi:
  HEADER    = '\033[95m'
  OKBLUE    = '\033[94m'
  OKGREEN   = '\033[92m'
  WARNING   = '\033[93m'
  FAIL      = '\033[91m'
  BOLD      = '\033[1m'
  UNDERLINE = '\033[4m'
  ENDC      = '\033[0m'

  @classmethod
  def _apply(cls, code: str, text: str, end: str = '') -> str:
    return f"{code}{text}{cls.ENDC}{end}"

  @classmethod
  def _print(cls, text: str) -> None:
    print(text.lstrip())

  @classmethod
  def info(cls, msg: str, emoji: str = '☑') -> None:
    cls._print(f"{emoji} {cls._apply(cls.OKBLUE, msg)}")

  @classmethod
  def success(cls, msg: str, emoji: str = '✅') -> None:
    cls._print(f"{emoji} {cls._apply(cls.OKGREEN, msg)}")

  @classmethod
  def warn(cls, msg, emoji: str = '⚠') -> None:
    cls._print(f"{emoji} {cls._apply(cls.WARNING, msg)}")

  @classmethod
  def error(cls, msg: str, emoji: str = '❌') -> None:
    cls._print(f"{emoji} {cls._apply(cls.FAIL, msg)}")

  @classmethod
  def header(cls, msg: str, emoji: str = '') -> None:
    cls._print(f"{emoji} {cls._apply(cls.HEADER, msg)}")

def git_clone(url: str, name: str, dst: str) -> None:
  repo_path  = join(dst, name)
  repo_exist = exists(repo_path)
  if not (repo_exist):
    ansi.warn(f"Git repo not found, cloning {name}", '📥')
    process = subprocess.run(['git', 'clone', '--depth=1', url, name],
        stdout=subprocess.PIPE, universal_newlines=True)
    if (process.stderr):
      ansi.error(f"Error while cloning {name}")
      ansi.error(process.stderr)
  else:
    ansi.success(f"Found Git repo {name}")

def chdir_if_needed(dst: str):
  full_path = abspath(dst)
  cwd = getcwd()
  if not (exists(full_path)):
    ansi.error(f"Could not find dir {full_path}")
  if (full_path == cwd):
    ansi.info(f"Already in dir {full_path}", "📁")
  else:
    chdir(full_path)
    ansi.info(f"Changed dirs into {full_path}", "📁")

def gdown_then_mv(url: str, f: str, dst: str):
  final_path = join(dst, f)

  if not (exists(final_path)):
    ansi.info(f"Downloading model {f}...", "📥")
    process = subprocess.run(['gdown', '-O', final_path, '-q', url],
                             stdout=subprocess.PIPE, universal_newlines=True)
    if (process.stderr):
      ansi.error(f"Error while downloading {url}")
      ansi.error(process.stderr)
      return
  else:
    ansi.success(f"Already downloaded model {f}", "📦")