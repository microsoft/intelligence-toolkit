# Define the URL of the .exe file to download
$url = "https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.msvc2015-win64.exe"

$folderPath = "app/wkhtmltox"
$outputFilePath = Join-Path -Path $folderPath -ChildPath "wkhtmltox-0.12.6-1.msvc2015-win64_.exe"

# Check if the file already exists in the folder
if (Test-Path $outputFilePath) {
    Write-Host "File already exists. Skipping download."
} else {

    Write-Host "Downloading wkhtmltox from host: $url"

    Invoke-WebRequest -Uri $url -OutFile $outputFilePath

    if (Test-Path $outputFilePath) {
        Write-Host "Download successful."
        # Call pynsist with the installer configuration file
    } else {
        Write-Host "Download failed. Exiting script."
        exit 1
    }
}

pynsist .\installer.cfg
