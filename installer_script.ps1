# Define the URL of the .exe file to download
$exeUrl = "https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox-0.12.6-1.msvc2015-win64.exe"
$wheelUrl = "https://github.com/dayesouza/python-louvain/raw/4be44746eca195eadc97e794b40984fd47420aa4/wheel/python_louvain-0.16-py3-none-any.whl"

$folderPath = "app/wkhtmltox"
if (-not (Test-Path $folderPath)) {
    New-Item -Path $folderPath -ItemType Directory
}
$outputFilePathExe = Join-Path -Path $folderPath -ChildPath "wkhtmltox-0.12.6-1.msvc2015-win64_.exe"


$wheelFolder = "wheels"
if (-not (Test-Path $wheelFolder)) {
    New-Item -Path $wheelFolder -ItemType Directory
}
$wheelFilePath = Join-Path -Path $wheelFolder -ChildPath "python_louvain-0.16-py3-none-any.whl"
# Check if the file already exists in the folder
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

if (Test-Path $wheelFilePath) {
    Write-Host "Wheel file already exists. Skipping download."
} else {

    Write-Host "Downloading python-louvain wheel from host: $wheelUrl"

    Invoke-WebRequest -Uri $wheelUrl -OutFile $wheelFilePath

    if (Test-Path $wheelFilePath) {
        Write-Host "Download of wheel successful."
    } else {
        Write-Host "Download failed. Exiting script."
        exit 1
    }
}
#split it here

#code sign
pynsist .\installer.cfg
