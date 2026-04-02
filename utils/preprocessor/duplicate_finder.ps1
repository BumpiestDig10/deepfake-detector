# Define the folder to search (change this to your target directory)
$targetFolder = "E:\deepfake-detector\instaProfiles\photos"

Write-Host "Scanning for duplicate files in: $targetFolder"

# Get all files recursively
$allFiles = Get-ChildItem -Path $targetFolder -Recurse -File

# Group files by their hash (this identifies duplicates)
$fileGroups = $allFiles | Group-Object -Property Length | Where-Object { $_.Count -gt 1 } | ForEach-Object {
    $_.Group | Group-Object -Property { (Get-FileHash $_.FullName -Algorithm MD5).Hash }
} | Where-Object { $_.Count -gt 1 }

# Display duplicates found
Write-Host "Found $($fileGroups.Count) groups of duplicate files"

# Process each group of duplicates
foreach ($group in $fileGroups) {
    Write-Host "`nDuplicate Group (MD5: $($group.Name))"
    $filesToKeep = $group.Group | Select-Object -First 1
    $filesToRemove = $group.Group | Select-Object -Skip 1
    
    Write-Host "Keeping: $($filesToKeep.FullName)"
    
    foreach ($file in $filesToRemove) {
        # Uncomment the next line to actually delete files
        Remove-Item -Path $file.FullName -Force
        Write-Host "Deleted: $($file.FullName)"
    }
}