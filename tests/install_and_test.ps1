# Edge Bookmark MCP Server - Installation and Testing Script
# Run this in PowerShell to set up and test the server

param(
    [switch]$SkipInstall,
    [switch]$TestOnly,
    [string]$ProjectPath = "D:\projects\windsurf\mcp\edge-bookmark-mcp-server"
)

Write-Host "🚀 Edge Bookmark MCP Server - Installation & Testing" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Function to check if command exists
function Test-CommandExists {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to run command with output capture
function Invoke-CommandWithOutput {
    param(
        [string]$Command,
        [string]$Arguments = "",
        [string]$WorkingDirectory = $PWD
    )
    
    $outputFile = "C:\temp\cmd_output_$(Get-Date -Format 'HHmmss').txt"
    
    try {
        if ($Arguments) {
            Start-Process -FilePath $Command -ArgumentList $Arguments -Wait -RedirectStandardOutput $outputFile -RedirectStandardError $outputFile -WindowStyle Hidden -WorkingDirectory $WorkingDirectory
        } else {
            Start-Process -FilePath $Command -Wait -RedirectStandardOutput $outputFile -RedirectStandardError $outputFile -WindowStyle Hidden -WorkingDirectory $WorkingDirectory
        }
        
        if (Test-Path $outputFile) {
            $output = Get-Content $outputFile -Encoding UTF8 -Raw
            Remove-Item $outputFile -ErrorAction SilentlyContinue
            return $output
        }
        return ""
    } catch {
        Write-Warning "Command execution failed: $($_.Exception.Message)"
        return ""
    }
}

# Step 1: Check Prerequisites
Write-Host "`n🔍 Checking Prerequisites..." -ForegroundColor Yellow

# Check Python
if (Test-CommandExists "python") {
    $pythonVersion = Invoke-CommandWithOutput "python" "--version"
    Write-Host "✅ Python found: $($pythonVersion.Trim())" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

# Check pip
if (Test-CommandExists "pip") {
    Write-Host "✅ pip found" -ForegroundColor Green
} else {
    Write-Host "❌ pip not found. Please ensure pip is installed." -ForegroundColor Red
    exit 1
}

# Check project directory
if (Test-Path $ProjectPath) {
    Write-Host "✅ Project directory found: $ProjectPath" -ForegroundColor Green
} else {
    Write-Host "❌ Project directory not found: $ProjectPath" -ForegroundColor Red
    Write-Host "Please create the project structure first or update the -ProjectPath parameter" -ForegroundColor Yellow
    exit 1
}

# Step 2: Create Virtual Environment
if (-not $SkipInstall -and -not $TestOnly) {
    Write-Host "`n🐍 Setting up Python Virtual Environment..." -ForegroundColor Yellow
    
    Set-Location $ProjectPath
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Gray
        $output = Invoke-CommandWithOutput "python" "-m venv venv" $ProjectPath
        if (Test-Path "venv") {
            Write-Host "✅ Virtual environment created" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
            Write-Host $output
            exit 1
        }
    } else {
        Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
    }
    
    # Activate virtual environment (for current session)
    $venvActivate = Join-Path $ProjectPath "venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        Write-Host "Activating virtual environment..." -ForegroundColor Gray
        & $venvActivate
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    }
}

# Step 3: Install Dependencies
if (-not $SkipInstall -and -not $TestOnly) {
    Write-Host "`n📦 Installing Dependencies..." -ForegroundColor Yellow
    
    # Use pip from virtual environment if available
    $pipCommand = if (Test-Path "$ProjectPath\venv\Scripts\pip.exe") { "$ProjectPath\venv\Scripts\pip.exe" } else { "pip" }
    
    # Install basic dependencies
    $dependencies = @(
        "fastmcp>=2.10.0",
        "rapidfuzz>=3.6.0",
        "aiofiles>=23.0.0",
        "psutil>=5.9.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0"
    )
    
    foreach ($dep in $dependencies) {
        Write-Host "Installing $dep..." -ForegroundColor Gray
        $output = Invoke-CommandWithOutput $pipCommand "install $dep" $ProjectPath
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $dep installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Warning: $dep installation may have issues" -ForegroundColor Yellow
            Write-Host $output -ForegroundColor Gray
        }
    }
    
    Write-Host "✅ Dependencies installation completed" -ForegroundColor Green
}

# Step 4: Create Project Files
Write-Host "`n📁 Setting up Project Files..." -ForegroundColor Yellow

Set-Location $ProjectPath

# Ensure directories exist
$directories = @("src", "tests", "examples", "data", "data\backups", "data\exports", "data\cache")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    }
}

# Create __init__.py files
$initFiles = @("src\__init__.py", "tests\__init__.py")
foreach ($initFile in $initFiles) {
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile -Force | Out-Null
        Write-Host "✅ Created: $initFile" -ForegroundColor Green
    }
}

# Create basic README if it doesn't exist
if (-not (Test-Path "README.md")) {
    $readmeContent = @"
# Edge Bookmark MCP Server

Microsoft Edge bookmark management server using FastMCP 2.10.1+

## Features
- 📖 Read & Parse Edge bookmarks from all profiles
- 🔍 Advanced fuzzy search with RapidFuzz
- 📊 Bookmark analytics and duplicate detection
- 📤 Export to JSON, CSV, HTML formats
- 🔄 Real-time file monitoring

## Installation
1. Run ``install_and_test.ps1`` in PowerShell
2. Configure Windsurf MCP settings
3. Start the server with ``python src\server.py``

## Testing
Run ``python test_server.py`` to test components.

Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
"@
    
    Set-Content -Path "README.md" -Value $readmeContent -Encoding UTF8
    Write-Host "✅ Created README.md" -ForegroundColor Green
}

# Step 5: Run Component Tests
Write-Host "`n🧪 Running Component Tests..." -ForegroundColor Yellow

# Check if test file needs to be created
$testFile = "test_server.py"
if (-not (Test-Path $testFile)) {
    Write-Host "⚠️ Test file not found. Please ensure test_server.py is in the project root." -ForegroundColor Yellow
    Write-Host "You can create it using the provided artifact." -ForegroundColor Gray
} else {
    Write-Host "Running component tests..." -ForegroundColor Gray
    
    # Use python from virtual environment if available
    $pythonCommand = if (Test-Path "$ProjectPath\venv\Scripts\python.exe") { "$ProjectPath\venv\Scripts\python.exe" } else { "python" }
    
    $testOutput = Invoke-CommandWithOutput $pythonCommand $testFile $ProjectPath
    
    if ($testOutput) {
        Write-Host $testOutput -ForegroundColor Gray
        
        # Check if tests passed
        if ($testOutput -match "All tests passed") {
            Write-Host "`n🎉 All component tests passed!" -ForegroundColor Green
        } elseif ($testOutput -match "tests passed") {
            Write-Host "`n⚠️ Some tests passed, check output above for details" -ForegroundColor Yellow
        } else {
            Write-Host "`n❌ Tests failed, check output above for errors" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ No test output received. Check for errors." -ForegroundColor Red
    }
}

# Step 6: Check Edge Bookmarks
Write-Host "`n📖 Checking Edge Bookmarks..." -ForegroundColor Yellow

$edgeProfilesBase = "$env:LOCALAPPDATA\Microsoft\Edge\User Data"
if (Test-Path $edgeProfilesBase) {
    Write-Host "✅ Edge data directory found: $edgeProfilesBase" -ForegroundColor Green
    
    # Check for bookmark files
    $bookmarkFiles = @()
    
    # Default profile
    $defaultBookmarks = Join-Path $edgeProfilesBase "Default\Bookmarks"
    if (Test-Path $defaultBookmarks) {
        $bookmarkFiles += $defaultBookmarks
        Write-Host "✅ Default profile bookmarks found" -ForegroundColor Green
    }
    
    # Additional profiles
    $profileDirs = Get-ChildItem -Path $edgeProfilesBase -Directory -Name "Profile *" -ErrorAction SilentlyContinue
    foreach ($profileDir in $profileDirs) {
        $profileBookmarks = Join-Path $edgeProfilesBase "$profileDir\Bookmarks"
        if (Test-Path $profileBookmarks) {
            $bookmarkFiles += $profileBookmarks
            Write-Host "✅ $profileDir bookmarks found" -ForegroundColor Green
        }
    }
    
    if ($bookmarkFiles.Count -gt 0) {
        Write-Host "📊 Found $($bookmarkFiles.Count) bookmark file(s)" -ForegroundColor Cyan
        
        # Show file sizes
        foreach ($file in $bookmarkFiles) {
            $size = (Get-Item $file).Length
            $profile = Split-Path (Split-Path $file -Parent) -Leaf
            Write-Host "   📄 $profile`: $([math]::Round($size/1KB, 1)) KB" -ForegroundColor Gray
        }
    } else {
        Write-Host "⚠️ No bookmark files found. You may need to create some bookmarks in Edge first." -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Edge data directory not found. Is Microsoft Edge installed?" -ForegroundColor Yellow
    Write-Host "Expected location: $edgeProfilesBase" -ForegroundColor Gray
}

# Step 7: Next Steps
Write-Host "`n🎯 Next Steps for Windsurf Integration:" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

Write-Host "1. 📝 Configure Windsurf MCP Settings:" -ForegroundColor White
Write-Host "   Add to your Windsurf MCP configuration:" -ForegroundColor Gray
Write-Host @"
   {
     "mcpServers": {
       "edge-bookmark-server": {
         "command": "python",
         "args": ["$ProjectPath\src\server.py"],
         "env": {
           "BOOKMARK_DEBUG_MODE": "true",
           "BOOKMARK_DATA_DIR": "$ProjectPath\data"
         }
       }
     }
   }
"@ -ForegroundColor DarkGray

Write-Host "`n2. 🧪 Test FastMCP Server:" -ForegroundColor White
Write-Host "   cd $ProjectPath" -ForegroundColor Gray
Write-Host "   python src\server.py" -ForegroundColor Gray

Write-Host "`n3. 🔧 Development Commands:" -ForegroundColor White
Write-Host "   # Run tests:" -ForegroundColor Gray
Write-Host "   python test_server.py" -ForegroundColor Gray
Write-Host "   # Check configuration:" -ForegroundColor Gray
Write-Host "   python -c `"from src.config import config; print(config.to_dict())`"" -ForegroundColor Gray

Write-Host "`n4. 📊 Monitor logs:" -ForegroundColor White
Write-Host "   tail -f edge-bookmark-server.log" -ForegroundColor Gray

# Step 8: Environment Summary
Write-Host "`n📋 Environment Summary:" -ForegroundColor Cyan
Write-Host "Project Path: $ProjectPath" -ForegroundColor Gray
Write-Host "Python: $(if (Test-Path "$ProjectPath\venv\Scripts\python.exe") { 'Virtual Environment' } else { 'System Python' })" -ForegroundColor Gray
Write-Host "Edge Profiles: $($bookmarkFiles.Count) found" -ForegroundColor Gray
Write-Host "Status: Ready for testing" -ForegroundColor Green

Write-Host "`n✅ Setup completed! You can now test the server components." -ForegroundColor Green
Write-Host "Run 'python test_server.py' to validate the implementation." -ForegroundColor White