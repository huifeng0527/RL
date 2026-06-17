$manifest = 'C:\Users\admin\Desktop\research\RL\logs\retrain_required\missing_experiments_20260603_223219\queue_manifest.json'
$monitorLog = 'C:\Users\admin\Desktop\research\RL\logs\retrain_required\runner_logs\monitor_20260603_223219.log'
while ($true) {
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  try {
    $data = Get-Content -Raw $manifest | ConvertFrom-Json
    $counts = $data | Group-Object status | ForEach-Object { "$($_.Name)=$($_.Count)" }
    $running = $data | Where-Object { $_.status -eq 'running' } | Select-Object -First 1
    $line = if ($running) { "$ts `t $($counts -join ', ') `t running=$($running.suite)/$($running.name)/seed_$($running.seed)" } else { "$ts `t $($counts -join ', ') `t running=none" }
    Add-Content -Path $monitorLog -Value $line
    if (($data | Where-Object { $_.status -in @('pending','running') }).Count -eq 0) { break }
  } catch {
    Add-Content -Path $monitorLog -Value "$ts `t monitor_error=$($_.Exception.Message)"
  }
  Start-Sleep -Seconds 300
}
