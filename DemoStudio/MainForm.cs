using System;
using System.Linq;
using System.Windows.Forms;
using SileroVAD;

namespace DemoStudio
{
    public partial class MainForm : Form
    {
        private readonly SileroProcessor _processor = new SileroProcessor(@"silero.onnx");

        public MainForm()
        {
            InitializeComponent();
        }

        private void SelectFileButton_Click(object sender, EventArgs e)
        {
            var dlg = new OpenFileDialog { Multiselect = false };
            if (dlg.ShowDialog() != DialogResult.OK) return;
            var segs = _processor.ProcessFile(dlg.FileName);

            listView1.Items.Clear();
            var counter = 1;
            foreach (var row in segs.Select(seg => new ListViewItem(new[]
                         { counter.ToString(), $@"{seg.StartSeconds}", $@"{seg.StopSeconds}", $@"{seg.Score}" })))
            {
                listView1.Items.Add(row);
                counter++;
            }
        }
    }
}