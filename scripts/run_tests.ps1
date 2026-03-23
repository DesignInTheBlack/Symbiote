param(
    [int]$TimeoutSec = 900
)

$argsList = @("test", "--workspace")
Write-Host "[tests] Running: cargo $($argsList -join ' ') (timeout ${TimeoutSec}s)"

$proc = Start-Process -FilePath "cargo" -ArgumentList $argsList -NoNewWindow -PassThru
if ($proc.WaitForExit($TimeoutSec * 1000)) {
    Write-Host "[tests] Completed with exit code $($proc.ExitCode)"
    exit $proc.ExitCode
}

Write-Error "[tests] Timeout exceeded (${TimeoutSec}s). Killing cargo test."
try {
    $proc.Kill()
} catch {}
exit 1
