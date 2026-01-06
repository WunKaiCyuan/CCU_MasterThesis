dotnet run --project ./CCU_Regulations_Downloader/CCU_Regulations_Downloader.csproj && \
cd ./DownloadedResources && \
textutil -convert docx *.doc && \
textutil -convert docx *.odt