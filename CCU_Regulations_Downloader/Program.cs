namespace CCU_Regulations_Downloader;

internal static class Program
{
    static void Main(string[] args)
    {
        var downloader = new Downloader();
        downloader.StartAsync().GetAwaiter().GetResult();
    }
}