# Define the URL of the .exe file to download
$exeUrl = "https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.msvc2015-win64.exe"

Write-Host "Init time: $(Get-Date)"

$folderPath = "app/wkhtmltox"
if (-not (Test-Path $folderPath)) {
    New-Item -Path $folderPath -ItemType Directory
}
$outputFilePathExe = Join-Path -Path $folderPath -ChildPath "wkhtmltox-0.12.6-1.msvc2015-win64_.exe"


if (Test-Path $outputFilePathExe) {
    Write-Host "File already exists. Skipping download."
} else {

    Write-Host "Downloading wkhtmltox from host: $exeUrl"

    Invoke-WebRequest -Uri $exeUrl -OutFile $outputFilePathExe

    if (Test-Path $outputFilePathExe) {
        Write-Host "Download of exe successful."
        # Call pynsist with the installer configuration file
    } else {
        Write-Host "Download failed. Exiting script."
        exit 1
    }
}

Write-Host "Copying readme file..."
Copy-Item .\README.md "app/README.md"

Write-Host "Creating installer...."

pynsist .\installer.cfg

#Delete readme file
Remove-Item "app/README.md"

# print end time
Write-Host "Done!"
Write-Host "End time: $(Get-Date)"
