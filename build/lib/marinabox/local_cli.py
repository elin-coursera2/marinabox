import click
from .local_manager import LocalContainerManager
import json
from datetime import datetime
from .config import Config
import asyncio
from .computer_use.cli import main as computer_use_main
from pathlib import Path
from dotenv import dotenv_values

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@click.group()
def local():
    """Local container management commands"""
    pass

@local.command()
@click.option('--env-type', type=click.Choice(['browser', 'desktop']), default="browser", help='Environment type')
@click.option('--resolution', default="1280x800x24", help='Screen resolution')
@click.option('--tag', help='Add a tag to the session')
@click.option('--mount', type=click.Path(exists=True, dir_okay=True, file_okay=False), help='Directory to mount into the container at /mnt/host')
@click.option('--kiosk', is_flag=True, default=False, help='Launch Chrome in kiosk mode (browser env only)')
def create(env_type, resolution, tag, mount, kiosk):
    """Create a new session"""
    manager = LocalContainerManager()
    session = manager.create_session(
        env_type=env_type,
        resolution=resolution,
        tag=tag,
        mount_path=mount,
        kiosk=kiosk
    )
    click.echo(json.dumps(session.__dict__, cls=DateTimeEncoder, indent=2))

@local.command()
def list():
    """List all active sessions"""
    manager = LocalContainerManager()
    sessions = manager.list_sessions()
    click.echo(json.dumps([s.to_dict() for s in sessions], cls=DateTimeEncoder, indent=2))

@local.command()
@click.argument('session_id')
def get(session_id):
    """Get details for a specific session"""
    manager = LocalContainerManager()
    session = manager.get_session(session_id)
    if session:
        click.echo(json.dumps(session.to_dict(), cls=DateTimeEncoder, indent=2))
    else:
        click.echo("Session not found", err=True)

@local.command()
@click.argument('session_id')
@click.option('--video-dir', help='Custom directory to store video recording')
@click.option('--video-filename', help='Custom filename for video recording')
def stop(session_id, video_dir, video_filename):
    """Stop a browser session"""
    manager = LocalContainerManager(
        videos_path=Path(video_dir) if video_dir else None
    )
    success = manager.stop_session(session_id, video_filename=video_filename)
    if success:
        click.echo("Session stopped successfully")
    else:
        click.echo("Failed to stop session", err=True)

@local.command()
@click.option('--video-dir', help='Custom directory to store video recording')
def stop_all(video_dir):
  manager = LocalContainerManager(
    videos_path=Path(video_dir) if video_dir else None
  )
  sessions = manager.list_sessions()
  all_succeeded = True
  for session in sessions:
    success = manager.stop_session(session.session_id)
    if not success:
      all_succeeded = False
      click.echo(f"Failed to stop session {session.session_id}", err=True)
  if all_succeeded:
    click.echo("All sessions stopped successfully")

@local.command()
def list_closed():
    """List all closed sessions"""
    manager = LocalContainerManager()
    sessions = manager.list_closed_sessions()
    click.echo(json.dumps([s.__dict__ for s in sessions], cls=DateTimeEncoder, indent=2))

@local.command()
@click.argument('session_id')
def get_closed(session_id):
    """Get details for a specific closed session"""
    manager = LocalContainerManager()
    session = manager.get_closed_session(session_id)
    if session:
        data = session.__dict__.copy()
        if session.video_path:
            data['video_path'] = str(session.video_path)
        click.echo(json.dumps(data, cls=DateTimeEncoder, indent=2))
    else:
        click.echo("Closed session not found", err=True)

@local.command()
@click.argument('session_id')
@click.argument('tag')
def tag(session_id, tag):
    """Add or update tag for a session"""
    manager = LocalContainerManager()
    session = manager.update_tag(session_id, tag)
    if session:
        click.echo(json.dumps(session.__dict__, cls=DateTimeEncoder, indent=2))
    else:
        click.echo("Session not found", err=True)

@local.command()
@click.option('--anthropic-api-key', required=True, help='Anthropic API key')
def set_anthropic_key(anthropic_api_key):
    """Set configuration values"""
    config = Config()
    config.set_anthropic_key(anthropic_api_key)
    click.echo("API key set successfully")

@local.command()
@click.option('--env-file', required=True, help='Path to environment file')
def set_aws_credentials(env_file):
  """Set configuration values"""
  env = dotenv_values(env_file)
  config = Config()
  config.set_aws_access_key_id(env['AWS_ACCESS_KEY_ID'])
  config.set_aws_secret_access_key(env['AWS_SECRET_ACCESS_KEY'])
  config.set_aws_session_token(env['AWS_SESSION_TOKEN'])
  config.set_aws_region(env['AWS_DEFAULT_REGION'])
  click.echo("AWS credentials set successfully")

@local.command()
@click.argument('session_identifier')
@click.option('--command', required=True, help='Command to execute')
def computer_use(session_identifier, command):
    """Execute computer use command on a session"""
    # Check for API key
    config = Config()
    # api_key = config.get_anthropic_key()
    # if not api_key:
    #     click.echo("Error: Anthropic API key not set. Use 'mb local set --anthropic-api-key' first", err=True)
    #     return
    aws_access_key_id = config.get_aws_access_key_id()
    aws_secret_key = config.get_aws_secret_access_key()
    aws_session_token = config.get_aws_session_token()
    aws_region = config.get_aws_region()

    # Get session by ID or tag
    manager = LocalContainerManager()
    session = None
    
    # Try by ID first
    session = manager.get_session(session_identifier)
    
    # If not found, try by tag
    if not session:
        sessions = manager.list_sessions()
        matching_sessions = [s for s in sessions if s.tag == session_identifier]
        if len(matching_sessions) == 1:
            session = matching_sessions[0]
        elif len(matching_sessions) > 1:
            click.echo("Error: Multiple sessions found with this tag", err=True)
            return
    
    if not session:
        click.echo("Error: No session found with this ID or tag", err=True)
        return

    
    # Execute computer use command
    responses = asyncio.run(computer_use_main(command, aws_access_key_id, aws_secret_key, aws_session_token, aws_region, session.computer_use_port))
