using HtmlAgilityPack;

namespace CCU_Regulations_Downloader;

internal class Downloader
{
    private readonly string baseUrl = "https://oaa.ccu.edu.tw";
    private readonly string  downloadFolder = "DownloadedResources";
    
    public async Task StartAsync()
    {
        if(Directory.Exists(downloadFolder)) Directory.Delete(downloadFolder);
        Directory.CreateDirectory(downloadFolder);
        
        using HttpClient client = new HttpClient();
        client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");

        var downloadPageLinks = await GetDownloadPageLinksAsync(client);

        var downloadResourceLinks = new List<DownloadResourceLink>();
        var pdfResourcesLinkTasks = downloadPageLinks
            .Select(downloadPageLink => GetPDFResourcesLinksAsync(client, downloadPageLink))
            .ToList();
        await Task.WhenAll(pdfResourcesLinkTasks);
        foreach (var task in pdfResourcesLinkTasks)
        {
            downloadResourceLinks.AddRange(task.Result);
        }

        var downloadTasks = downloadResourceLinks.Select(link =>
        {
            var savePath = Path.Combine(downloadFolder, link.AliasFileName);
            var downloadTask = DownloadPDFAsync(client, link.Link, savePath);
            return downloadTask;
        });
        await Task.WhenAll(downloadTasks.ToArray());
        Console.WriteLine("檔案已下載完畢");
        
        Console.WriteLine("Press any key to exit..."); 
        Console.ReadLine();
    }

    private async Task<IEnumerable<DownloadPageLink>> GetDownloadPageLinksAsync(HttpClient client)
    {
        var searchUrl = "https://oaa.ccu.edu.tw/p/404-1004-11995.php?Lang=zh-tw";
        var htmlContent = await client.GetStringAsync(searchUrl);

        // 解析 HTML 尋找 PDF 連結
        var doc = new HtmlDocument();
        doc.LoadHtml(htmlContent);

        // 尋找所有 class 為 list-group-item dropdown 的 li
        var nodes = doc.DocumentNode.SelectNodes("//li[contains(@class, 'list-group-item') and contains(@class, 'dropdown')]");
        var pageLinks = nodes.Select(n =>
            {
                var category = n.SelectSingleNode("./a").InnerText;
                var href = n.SelectSingleNode("./a").GetAttributeValue("href", "");

                if (href.StartsWith("http")) return new DownloadPageLink { Category = category.Trim(), Link = href };
                
                var fullUrl = baseUrl + href;
                return new DownloadPageLink { Category = category.Trim(), Link = fullUrl };
            })
            .ToList();

        Console.WriteLine($"找到 {pageLinks.Count} 個 校規分類連結。");
        return pageLinks;
    }
    
    private async Task<IEnumerable<DownloadResourceLink>> GetPDFResourcesLinksAsync(HttpClient client, DownloadPageLink downloadPageLink)
    {
        var htmlContent = await client.GetStringAsync(downloadPageLink.Link);

        // 解析 HTML 尋找 PDF 連結
        var doc = new HtmlDocument();
        doc.LoadHtml(htmlContent);

        // 尋找所有 href 結尾為 .pdf 的 <a> 標籤
        var pdfUrls = doc.DocumentNode.SelectNodes("//tr/td")
            .Where(n =>
            {
                try
                {
                    n.SelectSingleNode(".//a[@href]").GetAttributeValue("href", "");
                    return true;
                }
                catch (Exception e)
                {
                    return false;
                }
            })
            .Select(n =>
            {
                string href;
                href = n.SelectSingleNode(".//a[@href]").GetAttributeValue("href", "");
                var fileName = Path.GetFileName(href);
                var ext = Path.GetExtension(fileName);
                string text;
                try
                {
                    text = HtmlEntity.DeEntitize(n.InnerText).Split("\u00A0")[2].Trim();
                }
                catch (Exception e)
                {
                    text = n.SelectSingleNode(".//p").InnerText;
                }
                var aliasFileName = $"{downloadPageLink.Category}_{text}{ext}";
                
                Console.WriteLine($"aliasFileName {aliasFileName}, href {href}");
                if (href.StartsWith("http"))
                    return new DownloadResourceLink
                        { Link = href, OriginalFileName = fileName, AliasFileName = aliasFileName };
                
                var fullUrl = baseUrl + href;
                return new DownloadResourceLink
                    { Link = fullUrl, OriginalFileName = fileName, AliasFileName = aliasFileName };
            })
            .Distinct()
            .ToList();
        Console.WriteLine($"{downloadPageLink.Category} 找到 {pdfUrls.Count} 個 PDF 檔案連結。");
        
        return pdfUrls;
    }

    private async Task<bool> DownloadPDFAsync(HttpClient client, string pdfUrl, string savePath)
    {
        Console.WriteLine($"正在下載: {pdfUrl}...");

        var fileBytes = await client.GetByteArrayAsync(pdfUrl);
        await File.WriteAllBytesAsync(savePath, fileBytes);

        Console.WriteLine($"[成功] 已存至: {savePath}");
        return true;
    }
}

class DownloadPageLink
{
    public string Link { get; set; }
    public string Category { get; set; }
}

class DownloadResourceLink
{
    public string Link { get; set; }
    public string OriginalFileName { get; set; }
    public string AliasFileName { get; set; }
}