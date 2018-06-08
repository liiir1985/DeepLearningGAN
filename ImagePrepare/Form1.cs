using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;

namespace ImagePrepare
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (FD.ShowDialog() == DialogResult.OK)
            {
                var files = System.IO.Directory.GetFiles(FD.SelectedPath, "M_*.png");
                Image tmpImg = new Bitmap(150, 150, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                using (Graphics tmpG = Graphics.FromImage(tmpImg))
                {   
                    foreach (var i in files)
                    {
                        var path = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(i), "Output");
                        if (!System.IO.Directory.Exists(path))
                            System.IO.Directory.CreateDirectory(path);

                        using (Image img = Bitmap.FromFile(i))
                        {
                            Image output = new Bitmap(64, 64, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                            tmpG.Clear(Color.White);
                            tmpG.DrawImage(img, new Rectangle(7, 7, 136, 136), new RectangleF(0, 16, 136, 136), GraphicsUnit.Pixel);
                            for (int x = 0; x < 3; x++)
                            {
                                for (int y = 0; y < 3; y++)
                                {
                                    var oriX = -5 + x * 5 + 27;
                                    var oriY = -5 + y * 5 + 27;
                                    using (Graphics g = Graphics.FromImage(output))
                                    {
                                        g.DrawImage(tmpImg, new Rectangle(0, 0, 64, 64), new RectangleF(oriX, oriY, 96, 96), GraphicsUnit.Pixel);
                                    }
                                    var filename = $"{ System.IO.Path.GetFileNameWithoutExtension(i)}_{x}_{y}.jpg";
                                    output.Save(System.IO.Path.Combine(path, filename));
                                }
                            }
                        }
                    }
                }
            }
        }

        Image RotateImage(Image img, float rotationAngle)
        {
            // When drawing the returned image to a form, modify your points by 
            // (-(img.Width / 2) - 1, -(img.Height / 2) - 1) to draw for actual co-ordinates.

            //create an empty Bitmap image 
            Bitmap bmp = new Bitmap((img.Width * 2), (img.Height * 2));

            //turn the Bitmap into a Graphics object
            Graphics gfx = Graphics.FromImage(bmp);

            //set the point system origin to the center of our image
            gfx.TranslateTransform((float)bmp.Width / 2, (float)bmp.Height / 2);

            //now rotate the image
            gfx.RotateTransform(rotationAngle);

            //move the point system origin back to 0,0
            gfx.TranslateTransform(-(float)bmp.Width / 2, -(float)bmp.Height / 2);

            //set the InterpolationMode to HighQualityBicubic so to ensure a high
            //quality image once it is transformed to the specified size
            gfx.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

            //draw our new image onto the graphics object with its center on the center of rotation
            gfx.DrawImage(img, new PointF((img.Width / 2), (img.Height / 2)));

            //dispose of our Graphics object
            gfx.Dispose();

            //return the image
            return bmp;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (FD.ShowDialog() == DialogResult.OK)
            {
                var files = System.IO.Directory.GetFiles(FD.SelectedPath, "*.jpg");
                foreach (var i in files)
                {
                    var path = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(i), "Output");
                    if (!System.IO.Directory.Exists(path))
                        System.IO.Directory.CreateDirectory(path);

                    using (Image img = Bitmap.FromFile(i))
                    {
                        Image output = new Bitmap(64, 64, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                        using (Graphics g = Graphics.FromImage(output))
                        {
                            g.DrawImage(img, new Rectangle(0, 0, 64, 64), new RectangleF(0, 0, 96, 96), GraphicsUnit.Pixel);
                        }
                        var filename = $"{ System.IO.Path.GetFileNameWithoutExtension(i)}.jpg";
                        output.Save(System.IO.Path.Combine(path, filename));
                    }
                }
            }
        }
    }
}
