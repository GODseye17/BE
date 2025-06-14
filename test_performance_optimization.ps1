# PowerShell script to test performance optimizations

Write-Host "Vivum Performance Optimization Test" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray

# Test 1: With Embeddings (Full RAG)
Write-Host "`nFULL RAG MODE (with embeddings):" -ForegroundColor Green
Write-Host ("-" * 40) -ForegroundColor Gray

$stopwatch1 = [System.Diagnostics.Stopwatch]::StartNew()
$response1 = Invoke-RestMethod -Uri "http://localhost:8000/fetch-topic-data" -Method Post -ContentType "application/json" -Body '{
    "topic": "latest diabetes treatment guidelines",
    "max_results": 20,
    "create_embeddings": true
}'

$topic_id_full = $response1.topic_id
Write-Host "Topic ID: $topic_id_full" -ForegroundColor Yellow

# Wait for completion
while ($true) {
    $status = (Invoke-RestMethod -Uri "http://localhost:8000/topic/$topic_id_full/status" -Method Get).status
    if ($status -eq "completed") {
        $stopwatch1.Stop()
        Write-Host "Completed in: $($stopwatch1.Elapsed.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Green
        Write-Host "RAG queries enabled!" -ForegroundColor Green
        break
    }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 2
}

# Test 2: Without Embeddings (Metadata Only)
Write-Host "`n`nFAST METADATA MODE (no embeddings):" -ForegroundColor Yellow
Write-Host ("-" * 40) -ForegroundColor Gray

$stopwatch2 = [System.Diagnostics.Stopwatch]::StartNew()
$response2 = Invoke-RestMethod -Uri "http://localhost:8000/fetch-topic-data" -Method Post -ContentType "application/json" -Body '{
    "topic": "latest diabetes treatment guidelines",
    "max_results": 20,
    "create_embeddings": false
}'

$topic_id_fast = $response2.topic_id
Write-Host "Topic ID: $topic_id_fast" -ForegroundColor Yellow

# Wait for completion
while ($true) {
    $status = (Invoke-RestMethod -Uri "http://localhost:8000/topic/$topic_id_fast/status" -Method Get).status
    if ($status -eq "completed") {
        $stopwatch2.Stop()
        Write-Host "Completed in: $($stopwatch2.Elapsed.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Yellow
        Write-Host "RAG queries NOT available (metadata only)" -ForegroundColor DarkYellow
        break
    }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 1
}

# Summary
Write-Host "`n" + ("=" * 60) -ForegroundColor Gray
Write-Host "PERFORMANCE SUMMARY:" -ForegroundColor Cyan
Write-Host ("-" * 40) -ForegroundColor Gray

$time_full = $stopwatch1.Elapsed.TotalSeconds
$time_fast = $stopwatch2.Elapsed.TotalSeconds
$speedup = $time_full / $time_fast
$time_saved = $time_full - $time_fast

Write-Host "Full RAG Mode: $($time_full.ToString('F2'))s" -ForegroundColor White
Write-Host "Metadata Only: $($time_fast.ToString('F2'))s" -ForegroundColor White
Write-Host "Speedup: $($speedup.ToString('F1'))x faster" -ForegroundColor Green
Write-Host "Time Saved: $($time_saved.ToString('F2'))s" -ForegroundColor Green

# Quick test of RAG capability
Write-Host "`n`nTESTING RAG CAPABILITY:" -ForegroundColor Cyan
Write-Host ("-" * 40) -ForegroundColor Gray

# Try RAG query on full topic
try {
    $rag_test = Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body "{
        `"topic_id`": `"$topic_id_full`",
        `"query`": `"What are the key recommendations?`"
    }"
    Write-Host "RAG query successful on embeddings-enabled topic" -ForegroundColor Green
} catch {
    Write-Host "RAG query failed on embeddings-enabled topic" -ForegroundColor Red
}

# Try RAG query on metadata-only topic (should fail)
try {
    $rag_test2 = Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body "{
        `"topic_id`": `"$topic_id_fast`",
        `"query`": `"What are the key recommendations?`"
    }" -ErrorAction Stop
    Write-Host "Unexpected: RAG worked on metadata-only topic" -ForegroundColor Red
} catch {
    Write-Host "Correctly rejected RAG query on metadata-only topic" -ForegroundColor Green
}

Write-Host "`nPerformance test completed!" -ForegroundColor Cyan
