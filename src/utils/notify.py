"""System notification utility for Claude Code.

Usage:
    from src.utils.notify import send_notification
    send_notification("Training Complete", "Robot agent training finished!")
"""

import subprocess
import sys
import os


def send_notification(title: str, message: str, timeout: int = 5000):
    """Send a Windows balloon tip notification.

    Args:
        title: Notification title
        message: Notification body
        timeout: Display duration in milliseconds (default 5000)
    """
    if sys.platform != 'win32':
        print(f"[Notification] {title}: {message}")
        return

    # Write a temporary PS1 script to avoid escaping issues
    script_content = f'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notifyIcon = New-Object System.Windows.Forms.NotifyIcon
$icon = [System.Drawing.SystemIcons]::Information
$notifyIcon.Icon = $icon
$notifyIcon.Visible = $true
$notifyIcon.ShowBalloonTip({timeout}, "{title}", "{message}", "Info")
Start-Sleep -Seconds {max(6, timeout // 1000 + 1)}
$notifyIcon.Dispose()
'''

    script_path = os.path.join(os.environ.get('TEMP', 'C:\\Users\\admin\\AppData\\Local\\Temp'), 'claude_notify.ps1')

    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        subprocess.Popen(
            ['powershell', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', script_path],
            creationflags=0x08000000  # DETACHED_PROCESS
        )
    except Exception as e:
        print(f"[Notify] Error: {e}")
    finally:
        # Cleanup temp script after a delay
        if os.path.exists(script_path):
            try:
                os.remove(script_path)
            except:
                pass


def notification_on_complete(title_prefix: str = ""):
    """Decorator to send notification when a function completes.

    Usage:
        @notification_on_complete("Training")
        def long_running_task():
            # ... do work
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            title = f"{title_prefix} Complete" if title_prefix else "Task Complete"
            send_notification(title, f"{func.__name__} finished")
            return result
        return wrapper
    return decorator


if __name__ == '__main__':
    send_notification("Claude Code", "Notification system ready!")