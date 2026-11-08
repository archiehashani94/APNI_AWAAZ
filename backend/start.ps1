# Start the API on port 8001, replacing any server that already owns it.
#
# On Windows a second bind fails with WinError 10013 ("an attempt was made to
# access a socket in a way forbidden by its access permissions"), which looks
# like a firewall problem but is really just the port being taken.
#
#   .\start.ps1            # reload mode, for development
#   .\start.ps1 -NoReload  # plain server, for a stable session
param([int]$Port = 8001, [switch]$NoReload)

$existing = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
foreach ($conn in $existing) {
    Write-Host "Port $Port is held by PID $($conn.OwningProcess) - stopping it."
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
}
if ($existing) { Start-Sleep -Seconds 1 }

$args = @("-m", "uvicorn", "app.main:app", "--port", "$Port")
if (-not $NoReload) { $args += "--reload" }

& ".\.venv\Scripts\python.exe" @args
