param (
    [string]$Path = ".\datasets" # Default to "datasets" in the current directory if no path is provided
)
[string]$Path_raw  = "$Path\raw"
[string]$Path_full = "$Path\full"
[string]$Path_tmp = "$Path\_tmp"

function Download-Archive {
    param (
        [string]$Url_endpoint,
        [string]$OutputFileName,
        [string]$TempPath
    )
    [string] $Url= "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
    $Url = $Url+$Url_endpoint
    # Ensure the temp path exists
    if (-Not (Test-Path -Path $TempPath)) {
        New-Item -ItemType Directory -Path $TempPath | Out-Null
    }

    # Construct the full output path
    $OutputFilePath = "$TempPath\$OutputFileName"

    # Download the file
    Invoke-WebRequest -Uri $Url -OutFile $OutputFilePath
    Write-Host "Downloaded archive to '$OutputFilePath'."

    return $OutputFilePath
}

if (-Not (Test-Path -Path $Path)) {
    # Create the folder if it does not exist
    New-Item -ItemType Directory -Path $Path      | Out-Null
    New-Item -ItemType Directory -Path $Path_raw  | Out-Null
    New-Item -ItemType Directory -Path $Path_full | Out-Null
    Write-Host "Created directories at $Path."
    Download-Archive -Url_endpoint "mnist.scale.bz2"   -OutputFileName "mnist.scale.bz2"   -TempPath $Path_tmp | Out-Null
    Download-Archive -Url_endpoint "mnist.scale.t.bz2" -OutputFileName "mnist.scale.t.bz2" -TempPath $Path_tmp | Out-Null
    
    Write-Host "Uncompressing..."
    bunzip2 -ck "$Path_tmp\mnist.scale.bz2" > "$Path_raw\mnist.scale"
    bunzip2 -ck "$Path_tmp\mnist.scale.t.bz2" > "$Path_raw\mnist.scale.t"

    Write-Host "Joining..."
    cat "$Path_raw\mnist.scale" > "$Path_full\mnist.scale.full"
    cat "$Path_raw\mnist.scale.t" >> "$Path_full\mnist.scale.full"

    Remove-Item -Path $Path_tmp -Recurse -Force
    Write-Host "Datasets built successfully."
} else {
    Write-Host "Error: Folder '$Path' already exists."
}
