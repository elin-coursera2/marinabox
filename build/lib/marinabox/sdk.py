from typing import List, Optional, Dict
from .local_manager import LocalContainerManager
from .models import BrowserSession
from .config import Config
import asyncio
from .computer_use.cli import main as computer_use_main
from pathlib import Path

class MarinaboxSDK:
    def __init__(self, videos_path: Optional[str] = None):
        self.manager = LocalContainerManager(
            videos_path=Path(videos_path) if videos_path else None
        )
        self.config = Config()

    def create_session(
        self, 
        env_type: str = "browser", 
        resolution: str = "1280x800x24",
        tag: Optional[str] = None,
        kiosk: bool = False
    ) -> BrowserSession:
        """
        Create a new Marinabox session.
        
        Args:
            env_type: Either 'browser' or 'desktop'
            resolution: Screen resolution (e.g., '1280x800x24')
            tag: Optional tag for the session
            kiosk: Whether to launch Chrome in kiosk mode
            
        Returns:
            BrowserSession object
        """
        return self.manager.create_session(
            env_type=env_type, 
            resolution=resolution, 
            tag=tag,
            kiosk=kiosk
        )

    def list_sessions(self) -> List[BrowserSession]:
        """List all active sessions"""
        return self.manager.list_sessions()

    def get_session(self, session_id: str) -> Optional[BrowserSession]:
        """Get details for a specific session"""
        return self.manager.get_session(session_id)

    def stop_session(self, session_id: str, video_filename: Optional[str] = None) -> bool:
        """
        Stop a session
        
        Args:
            session_id: ID of the session to stop
            video_filename: Optional custom filename for the video recording
        """
        return self.manager.stop_session(session_id, video_filename=video_filename)

    def list_closed_sessions(self) -> List[BrowserSession]:
        """List all closed sessions"""
        return self.manager.list_closed_sessions()

    def get_closed_session(self, session_id: str) -> Optional[BrowserSession]:
        """Get details for a specific closed session"""
        return self.manager.get_closed_session(session_id)

    def update_tag(self, session_id: str, tag: str) -> Optional[BrowserSession]:
        """Add or update tag for a session"""
        return self.manager.update_tag(session_id, tag)

    def set_anthropic_key(self, api_key: str) -> None:
        """Set Anthropic API key"""
        self.config.set_anthropic_key(api_key)

    def set_aws_access_key_id(self, id: str):
      self.config.set_aws_access_key_id(id)

    def set_aws_secret_access_key(self, key: str):
      self.config.set_aws_secret_access_key(key)

    def set_aws_session_token(self, token: str):
      self.config.set_aws_session_token(token)

    def set_aws_region(self, region: str):
      self.config.set_aws_region(region)

    def get_session_by_identifier(self, identifier: str) -> Optional[BrowserSession]:
        """
        Get a session by ID or tag.
        
        Args:
            identifier: Session ID or tag
            
        Returns:
            BrowserSession object if found, None otherwise
        """
        # Try by ID first
        session = self.get_session(identifier)
        if session:
            return session

        # Try by tag
        sessions = self.list_sessions()
        matching_sessions = [s for s in sessions if s.tag == identifier]
        if len(matching_sessions) == 1:
            return matching_sessions[0]
        return None

    async def execute_computer_use_command(
        self, 
        session_identifier: str, 
        command: str
    ) -> List:
        """
        Execute a computer use command on a session.
        
        Args:
            session_identifier: Session ID or tag
            command: Command to execute
        
        Returns:
            List of response tuples containing the output
        
        Raises:
            ValueError: If API key is not set or session is not found
        """
        aws_access_key_id = self.config.get_aws_access_key_id()
        aws_secret_access_key = self.config.get_aws_secret_access_key()
        aws_session_token = self.config.get_aws_session_token()
        aws_region = self.config.get_aws_region()

        session = self.get_session_by_identifier(session_identifier)
        if not session:
            raise ValueError("No session found with this ID or tag")

        responses = await computer_use_main(command, aws_access_key_id, aws_secret_access_key, aws_session_token, aws_region, session.computer_use_port)
        return responses

    def computer_use_command(self, session_identifier: str, command: str) -> List:
        """
        Synchronous wrapper for execute_computer_use_command
        """
        responses = asyncio.run(self.execute_computer_use_command(session_identifier, command)) 
        return responses