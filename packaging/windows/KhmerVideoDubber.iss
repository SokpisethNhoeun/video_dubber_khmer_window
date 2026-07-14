#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#define MyAppName "Khmer Video Dubber"
#define MyAppExeName "KhmerVideoDubber.exe"

[Setup]
AppId={{A33A7C25-F4E8-4BA7-9271-14513129D144}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=NhoeunSokpiseth
DefaultDirName={autopf}\Khmer Video Dubber
DefaultGroupName={#MyAppName}
OutputDir=release
OutputBaseFilename=KhmerVideoDubber-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no

[Files]
Source: "..\..\dist\KhmerVideoDubber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
