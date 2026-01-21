# Qt Image Enhancer

A cross-platform image enhancement application built with QtPy for maximum compatibility across Qt backends.

## Features

- **Cross-platform compatibility** using QtPy (works with PyQt5/6 and PySide2/6)
- **Advanced Image Enhancement** powered by OpenCV:
  - Brightness, contrast, and saturation adjustment
  - Gaussian blur filter
  - Image sharpening
  - Edge detection
  - Emboss effect
  - Automatic contrast enhancement (histogram equalization)
- **File Operations**:
  - Load images (PNG, JPEG, BMP, TIFF)
  - Save enhanced images
- **Real-time preview** of enhancements
- **Professional interface** with organized control panels

## Requirements

- Python 3.8+
- QtPy
- PySide6 (or PyQt5/6/PySide2)
- OpenCV (cv2) for advanced image processing
- Pillow (PIL) for additional image operations
- NumPy

## Installation

1. **Clone/Download** the project to your local machine

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Windows Setup

Follow these steps on Windows using PowerShell or Command Prompt.

1. Install Python 3.8+ from https://www.python.org and select "Add Python to PATH" during installation.
2. Open Windows PowerShell (recommended) or Command Prompt.
3. Create and activate a virtual environment, install dependencies, and run:

PowerShell:
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
python main.py
```

Command Prompt (CMD):
```bat
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py
```

Notes:
- If activation fails with an execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
- If `pip` isn't recognized, use `python -m pip install -r requirements.txt`.
- Qt backend is provided by PySide6 (already included in requirements).

## Running the Application

```bash
python main.py
```

## Project Structure

```
QtImageEnhencer/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py  # Main window UI
│   └── core/
│       ├── __init__.py
│       └── image_processor.py  # Image processing logic
└── README.md
```

## Usage

1. **Load an Image**: Click "Load Image" and select an image file
2. **Enhance**: Use the sliders to adjust brightness, contrast, and saturation
3. **Preview**: Changes are applied in real-time
4. **Save**: Click "Save Image" to save your enhanced image
5. **Reset**: Click "Reset All" to return to original settings

## Development

### Adding New Features

- **New enhancement filters**: Add methods to `ImageProcessor` class
- **UI improvements**: Modify `MainWindow` class
- **Additional file formats**: Update file dialogs in the main window

### QtPy Advantages

This project uses QtPy as an abstraction layer, which means:
- **Automatic backend detection**: Works with whatever Qt binding you have installed
- **Easy migration**: Switch between PyQt and PySide without code changes
- **Future-proof**: Compatible with newer Qt versions as they're released

## Troubleshooting

### Common Issues

1. **"No module named 'qtpy'"**: Install QtPy with `pip install QtPy`
2. **Qt backend not found**: Install a Qt binding like `pip install PySide6`
3. **Image loading errors**: Ensure Pillow is installed: `pip install Pillow`

### Platform-Specific Notes

- **macOS**: May require additional permissions for file access
- **Windows**: Ensure proper PATH setup for Python and pip
- **Linux**: May need additional system packages for Qt

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by:
- Adding new image enhancement features
- Improving the user interface
- Adding support for more file formats
- Optimizing performance
- Writing tests