#!/usr/bin/env python3
import bs4
import sys
import argparse
import shlex

import fabric # SSH frontend
import paramiko # SSH backend, since it seems Fabric doesn't reexport AuthenticationException
import invoke # Command backend, again since exceptions aren't reexported

def die(message, status=1):
	if message is not None:
		print(message, file=sys.stderr)
	sys.exit(status)

def escape(x):
	return " ".join(shlex.quote(w) for w in x)

def get_gpu(conn):
	""" Ensures that the gpu isn't already busy, and returns its id"""
	nvi = bs4.BeautifulSoup(conn.run("nvidia-smi -q -x", hide='out').stdout, features="lxml-xml")
	# bs4 is really a bit of a pain
	nvi = nvi.find("nvidia_smi_log", recursive=False)
	try:
		gpus = [gpu] = nvi.find_all("gpu", recursive=False)
	except ValueError:
		die("Currently don't know how to handle multiple GPUs: " + ", ".join(gpu["id"] for gpu in gpus))
	else:
		for proc in gpu.processes.find_all("process_info", recursive=False):
			if proc.type.text != "G":
				cmdline = conn.run("cat /proc/%s/cmdline" % proc.pid.text, hide='out').stdout
				die("A computation is already running: %s" % escape(cmdline.split("\0")))
		return gpu["id"]

def open_session(server, password=None):
	""" Opens a connection, with an appropriate error if password is incorrect """
	try:
		conn = fabric.Connection(server, connect_kwargs={"password": password})
		conn.open()
		return conn
	except paramiko.AuthenticationException:
		die("Incorrect password" if password is not None else "Password required (-p *****)")

def main():
	argp = argparse.ArgumentParser()
	argp.add_argument("-H", "--host", default="sandal@129.16.225.95") # Protip: add "129.16.225.95 sandal" to /etc/hosts
	argp.add_argument("-p", "--password", default=None)
	argp.add_argument("-r", "--repo", default="https://github.com/elias-sundqvist/DATX02-20-04")
	argp.add_argument("-s", "--session", default=None)
	argp.add_argument("branch")
	argp.add_argument("command", nargs="+")
	args = argp.parse_args()

	with open_session(args.host, args.password) as conn:
		get_gpu(conn)
		try:
			conn.run("""
if tmux has -t %(session)s &>/dev/null; then
	echo Session "'"%(session)s"'" already running > /dev/stderr
	exit 1
fi
test -d repos || mkdir repos; cd repos
if test -d deploy; then
	cd deploy
	if [[ "$(git remote get-url origin)" != %(repo)s ]]; then
		echo 'repos/deploy has wrong remote!' > /dev/stderr
		exit 1
	fi
	git pull || exit 1
else
	git clone %(repo)s deploy || exit 1
	cd deploy
fi
git checkout %(branch)s || exit 1
tmux new-session -d -s %(session)s -- %(command)s || exit 1
echo Running "'"%(command)s"'" under session "'"%(session)s"'"
echo Attach with "'tmux attach -t "%(session)s"'"
			""" % {
				"repo": shlex.quote(args.repo),
				"branch": shlex.quote(args.branch),
				"command": escape(args.command),
				"session": shlex.quote(args.branch if args.session is None else args.session),
			})
		except invoke.UnexpectedExit:
			die(None)

if __name__ == "__main__":
	main()
