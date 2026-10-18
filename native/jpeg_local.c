/* Local JPEG coefficient editing. libjpeg-turbo 3.x, Windows x64.
 * Binary manifest: PJ01, u32 count, then u32 x,y,w,h,pathBytes,maskBytes;
 * UTF-8 path bytes, followed by one 0/1 byte per MCU in row-major order.
 * All file creation is exclusive. A warning is fatal: truncated JPEGs cannot
 * be silently repaired and published. The Python owner publishes atomically.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <bcrypt.h>
#include <shellapi.h>
#include <io.h>
#include <fcntl.h>
#include "jpeglib.h"

typedef struct { uint32_t x,y,w,h,n; char *path; unsigned char *mask; } Patch;
static Patch *patches; static uint32_t count;
static void fail(const char *message) { fprintf(stderr,"JPEG: %s\n",message); exit(2); }
static void *allocate(size_t n) { void *p=calloc(1,n?n:1); if(!p)fail("memory allocation"); return p; }
static wchar_t *wide(const char *p) {
 int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,p,-1,NULL,0);
 if(!n || n>32760)fail("invalid UTF-8 path");
 wchar_t *w=allocate((size_t)n*sizeof(wchar_t));
 if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,p,-1,w,n))fail("invalid UTF-8 path");
 return w;
}
static FILE *read_file(const char *p) { wchar_t *w=wide(p); FILE *f=_wfopen(w,L"rb"); free(w); if(!f)fail("cannot read file"); return f; }
static FILE *create_file(const char *p) {
 wchar_t *w=wide(p); HANDLE h=CreateFileW(w,GENERIC_WRITE,0,NULL,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL); free(w);
 if(h==INVALID_HANDLE_VALUE)fail("output exists or cannot be created");
 int fd=_open_osfhandle((intptr_t)h,_O_BINARY); if(fd<0){CloseHandle(h);fail("output descriptor");}
 FILE *f=_fdopen(fd,"wb"); if(!f){_close(fd);fail("output stream");} return f;
}
static void close_output(FILE *f) { if(fflush(f)||_commit(_fileno(f))||fclose(f))fail("output flush"); }
static void read_exact(FILE *f,void *p,size_t n) { if(fread(p,1,n,f)!=n)fail("truncated manifest or raw patch"); }
static uint32_t u32(FILE *f) { unsigned char p[4]; read_exact(f,p,4); return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24; }
static void manifest(const char *path) {
 FILE *f=read_file(path); char magic[4]; read_exact(f,magic,4); if(memcmp(magic,"PJ01",4))fail("manifest version");
 count=u32(f); if(count>65536)fail("manifest count"); patches=allocate((size_t)count*sizeof(Patch));
 for(uint32_t k=0;k<count;k++){
  Patch *p=&patches[k]; p->x=u32(f);p->y=u32(f);p->w=u32(f);p->h=u32(f);uint32_t len=u32(f);p->n=u32(f);
  if(p->x>65535||p->y>65535||!p->w||!p->h||p->w>256||p->h>256||!len||len>98300||!p->n||p->n>1024)fail("manifest limits");
  p->path=allocate((size_t)len+1);read_exact(f,p->path,len);if(memchr(p->path,0,len))fail("manifest NUL path");free(wide(p->path));
  p->mask=allocate(p->n);read_exact(f,p->mask,p->n);int touched=0;
  for(uint32_t j=0;j<p->n;j++){if(p->mask[j]>1)fail("manifest mask value");touched|=p->mask[j];}if(!touched)fail("empty manifest patch");
  for(uint32_t j=0;j<k;j++){Patch *q=&patches[j];if(p->x<q->x+q->w && q->x<p->x+p->w && p->y<q->y+q->h && q->y<p->y+p->h)fail("overlapping manifest tiles");}
 }
 if(fgetc(f)!=EOF||ferror(f))fail("trailing manifest content");
 fclose(f);
}
static void jpeg_error(j_common_ptr c) { (*c->err->output_message)(c); fail("invalid JPEG"); }
static void jpeg_message(j_common_ptr c,int level) { if(level<0)jpeg_error(c); }
static void errors(struct jpeg_error_mgr *e) { jpeg_std_error(e);e->error_exit=jpeg_error;e->emit_message=jpeg_message; }
static FILE *load(struct jpeg_decompress_struct *d,struct jpeg_error_mgr *e,const char *path,int markers) {
 memset(d,0,sizeof(*d));errors(e);d->err=e;jpeg_create_decompress(d);FILE *f=read_file(path);jpeg_stdio_src(d,f);
 if(markers){for(int m=0;m<16;m++)jpeg_save_markers(d,JPEG_APP0+m,65535);jpeg_save_markers(d,JPEG_COM,65535);}
 if(jpeg_read_header(d,TRUE)!=JPEG_HEADER_OK)fail("JPEG header");
 return f;
}
static void supported(struct jpeg_decompress_struct *d) {
 if(d->data_precision!=8||d->progressive_mode||d->arith_code||d->num_components!=3 || (d->jpeg_color_space!=JCS_RGB&&d->jpeg_color_space!=JCS_YCbCr))fail("unsupported JPEG encoding");
 int is444=d->max_h_samp_factor==1&&d->max_v_samp_factor==1;
 int is420=d->jpeg_color_space==JCS_YCbCr&&d->comp_info[0].h_samp_factor==2&&d->comp_info[0].v_samp_factor==2;
 for(int c=1;c<3;c++)is420=is420&&d->comp_info[c].h_samp_factor==1&&d->comp_info[c].v_samp_factor==1;
 if(!is444&&!is420)fail("unsupported JPEG sampling");
 for(int c=0;c<3;c++){int q=d->comp_info[c].quant_tbl_no;if(q<0||q>=NUM_QUANT_TBLS||!d->quant_tbl_ptrs[q])fail("missing quantization table");}
}
static void bounds(struct jpeg_decompress_struct *d) {
 unsigned mw=8*d->max_h_samp_factor,mh=8*d->max_v_samp_factor;
 for(uint32_t k=0;k<count;k++){Patch *p=&patches[k];
  if(p->x%mw||p->y%mh||p->x+p->w>d->image_width||p->y+p->h>d->image_height||
     (p->w%mw&&p->x+p->w!=d->image_width)||(p->h%mh&&p->y+p->h!=d->image_height)||
     p->n!=((p->w+mw-1)/mw)*((p->h+mh-1)/mh))fail("manifest MCU bounds");
 }
}
static int occupied(Patch *p,unsigned x,unsigned y,unsigned mw,unsigned mh) { return p->mask[(y/mh)*((p->w+mw-1)/mw)+x/mw]; }
static void row_mask(struct jpeg_decompress_struct *d,int c,unsigned y,unsigned char *mask) {
 jpeg_component_info *ci=&d->comp_info[c];memset(mask,0,ci->width_in_blocks);
 unsigned mw=8*d->max_h_samp_factor,mh=8*d->max_v_samp_factor;
 for(uint32_t k=0;k<count;k++){Patch *p=&patches[k];unsigned by=p->y/mh*ci->v_samp_factor,bx=p->x/mw*ci->h_samp_factor;
  unsigned rows=(p->h*ci->v_samp_factor+mh-1)/mh,cols=(p->w*ci->h_samp_factor+mw-1)/mw;
  if(y<by||y>=by+rows)continue;
  for(unsigned x=0;x<cols;x++)if(occupied(p,x/ci->h_samp_factor*mw,(y-by)/ci->v_samp_factor*mh,mw,mh))mask[bx+x]=1;
 }
}
static void hash_coefficients(struct jpeg_decompress_struct *d,jvirt_barray_ptr *arrays) {
 BCRYPT_ALG_HANDLE alg;DWORD objlen,n;if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,NULL,0)<0 || BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,(PUCHAR)&objlen,sizeof(objlen),&n,0)<0)fail("SHA256 initialization");
 printf("{\"width\":%u,\"height\":%u,\"coefficients\":[",d->image_width,d->image_height);
 for(int c=0;c<3;c++){
  BCRYPT_HASH_HANDLE hash;unsigned char *obj=allocate(objlen),digest[32];uint64_t outside=0;
  if(BCryptCreateHash(alg,&hash,obj,objlen,NULL,0,0)<0)fail("SHA256 create");
  jpeg_component_info *ci=&d->comp_info[c];unsigned char *mask=allocate(ci->width_in_blocks);
  for(unsigned y=0;y<ci->height_in_blocks;y++){
   row_mask(d,c,y,mask);JBLOCKARRAY row=(*d->mem->access_virt_barray)((j_common_ptr)d,arrays[c],y,1,FALSE);
   for(unsigned x=0;x<ci->width_in_blocks;){while(x<ci->width_in_blocks&&mask[x])x++;unsigned start=x;while(x<ci->width_in_blocks&&!mask[x])x++;
    if(x>start){if(BCryptHashData(hash,(PUCHAR)&row[0][start],(x-start)*sizeof(JBLOCK),0)<0)fail("SHA256 update");outside+=x-start;}}
  }
  if(BCryptFinishHash(hash,digest,32,0)<0)fail("SHA256 finish");
  BCryptDestroyHash(hash);free(obj);free(mask);
  if(c)printf(",");
  printf("{\"component\":%d,\"width_blocks\":%u,\"height_blocks\":%u,\"outside_blocks\":%llu,\"outside_sha256\":\"",c,ci->width_in_blocks,ci->height_in_blocks,(unsigned long long)outside);
  for(int i=0;i<32;i++)printf("%02x",digest[i]);
  printf("\"}");
 }printf("]}\n");BCryptCloseAlgorithmProvider(alg,0);
}
static void check_patch(struct jpeg_decompress_struct *s,struct jpeg_decompress_struct *p,Patch *r) {
 supported(p);if(p->image_width!=r->w||p->image_height!=r->h||s->jpeg_color_space!=p->jpeg_color_space)fail("patch dimensions/colorspace");
 for(int c=0;c<3;c++){jpeg_component_info *a=&s->comp_info[c],*b=&p->comp_info[c];
  if(a->component_id!=b->component_id||a->h_samp_factor!=b->h_samp_factor||a->v_samp_factor!=b->v_samp_factor||a->quant_tbl_no!=b->quant_tbl_no)fail("patch component/sampling/table mismatch");
  if(memcmp(s->quant_tbl_ptrs[a->quant_tbl_no]->quantval,p->quant_tbl_ptrs[b->quant_tbl_no]->quantval,64*sizeof(UINT16)))fail("patch quantization mismatch");}
}
static void encode_patches(const char *source) {
 struct jpeg_decompress_struct s;struct jpeg_error_mgr se;FILE *sf=load(&s,&se,source,0);supported(&s);bounds(&s);
 for(uint32_t k=0;k<count;k++){Patch *p=&patches[k];size_t len=strlen(p->path);char *raw=allocate(len+5);memcpy(raw,p->path,len);memcpy(raw+len,".rgb",5);FILE *rf=read_file(raw);free(raw);
  struct jpeg_compress_struct d;memset(&d,0,sizeof(d));struct jpeg_error_mgr de;errors(&de);d.err=&de;jpeg_create_compress(&d);jpeg_copy_critical_parameters(&s,&d);
  d.image_width=p->w;d.image_height=p->h;d.in_color_space=JCS_RGB;d.input_components=3;d.dct_method=JDCT_ISLOW;d.optimize_coding=FALSE;
  FILE *of=create_file(p->path);jpeg_stdio_dest(&d,of);jpeg_start_compress(&d,TRUE);unsigned char *row=allocate(p->w*3);
  for(unsigned y=0;y<p->h;y++){read_exact(rf,row,p->w*3);JSAMPROW rp=row;if(jpeg_write_scanlines(&d,&rp,1)!=1)fail("patch encode");}
  if(fgetc(rf)!=EOF||ferror(rf))fail("raw patch size mismatch");
  free(row);fclose(rf);jpeg_finish_compress(&d);jpeg_destroy_compress(&d);close_output(of);
 }jpeg_destroy_decompress(&s);fclose(sf);printf("{\"encoded_patches\":%u}\n",count);
}
static void patch_file(const char *source,const char *output) {
 struct jpeg_decompress_struct s;struct jpeg_error_mgr se;FILE *sf=load(&s,&se,source,1);supported(&s);bounds(&s);jvirt_barray_ptr *a=jpeg_read_coefficients(&s);
 unsigned mw=8*s.max_h_samp_factor,mh=8*s.max_v_samp_factor;
 for(uint32_t k=0;k<count;k++){Patch *r=&patches[k];struct jpeg_decompress_struct p;struct jpeg_error_mgr pe;FILE *pf=load(&p,&pe,r->path,0);check_patch(&s,&p,r);jvirt_barray_ptr *b=jpeg_read_coefficients(&p);
  for(int c=0;c<3;c++){jpeg_component_info *ci=&s.comp_info[c],*pi=&p.comp_info[c];unsigned bx=r->x/mw*ci->h_samp_factor,by=r->y/mh*ci->v_samp_factor;
   for(unsigned y=0;y<pi->height_in_blocks;y++){JBLOCKARRAY sr=(*s.mem->access_virt_barray)((j_common_ptr)&s,a[c],by+y,1,TRUE),pr=(*p.mem->access_virt_barray)((j_common_ptr)&p,b[c],y,1,FALSE);
    for(unsigned x=0;x<pi->width_in_blocks;x++)if(occupied(r,x/ci->h_samp_factor*mw,y/ci->v_samp_factor*mh,mw,mh))memcpy(sr[0][bx+x],pr[0][x],sizeof(JBLOCK));}
  }jpeg_finish_decompress(&p);jpeg_destroy_decompress(&p);fclose(pf);
 }
 struct jpeg_compress_struct d;memset(&d,0,sizeof(d));struct jpeg_error_mgr de;errors(&de);d.err=&de;jpeg_create_compress(&d);jpeg_copy_critical_parameters(&s,&d);d.write_JFIF_header=FALSE;d.write_Adobe_marker=FALSE;d.optimize_coding=FALSE;
 FILE *of=create_file(output);jpeg_stdio_dest(&d,of);jpeg_write_coefficients(&d,a);
 for(jpeg_saved_marker_ptr m=s.marker_list;m;m=m->next)jpeg_write_marker(&d,m->marker,m->data,m->data_length);
 jpeg_finish_compress(&d);jpeg_destroy_compress(&d);close_output(of);jpeg_finish_decompress(&s);jpeg_destroy_decompress(&s);fclose(sf);printf("{\"patched\":true,\"patches\":%u}\n",count);
}
static void pixels(const char *source,const char *output,int margin) {
 struct jpeg_decompress_struct a,b;struct jpeg_error_mgr ae,be;FILE *af=load(&a,&ae,source,0),*bf=load(&b,&be,output,0);supported(&a);supported(&b);bounds(&a);
 if(a.image_width!=b.image_width||a.image_height!=b.image_height)fail("pixel dimensions differ");
 a.out_color_space=b.out_color_space=JCS_RGB;a.dct_method=b.dct_method=JDCT_ISLOW;jpeg_start_decompress(&a);jpeg_start_decompress(&b);
 unsigned char *ar=allocate(a.output_width*3),*br=allocate(b.output_width*3),*mask=allocate(a.output_width);uint64_t diff=0,changed=0,outside=0;
 unsigned mw=8*a.max_h_samp_factor,mh=8*a.max_v_samp_factor;
 while(a.output_scanline<a.output_height){unsigned y=a.output_scanline;JSAMPROW ap=ar,bp=br;jpeg_read_scanlines(&a,&ap,1);jpeg_read_scanlines(&b,&bp,1);memset(mask,0,a.output_width);
  for(uint32_t k=0;k<count;k++){Patch *p=&patches[k];if((int)y<(int)p->y-margin||(int)y>=(int)(p->y+p->h)+margin)continue;unsigned cols=(p->w+mw-1)/mw,rows=(p->h+mh-1)/mh;
   for(unsigned my=0;my<rows;my++){int yy=p->y+my*mh;if((int)y<yy-margin||(int)y>=yy+(int)mh+margin)continue;
    for(unsigned mx=0;mx<cols;mx++)if(p->mask[my*cols+mx]){int x0=p->x+mx*mw-margin,x1=p->x+(mx+1)*mw+margin;if(x0<0)x0=0;if(x1>(int)a.output_width)x1=a.output_width;memset(mask+x0,1,x1-x0);}}
  }
  for(unsigned x=0;x<a.output_width;x++){int any=0;for(int c=0;c<3;c++)if(ar[3*x+c]!=br[3*x+c]){any=1;if(!mask[x])diff++;}changed+=any;if(!mask[x])outside++;}
 }jpeg_finish_decompress(&a);jpeg_finish_decompress(&b);jpeg_destroy_decompress(&a);jpeg_destroy_decompress(&b);fclose(af);fclose(bf);free(ar);free(br);free(mask);
 printf("{\"full_decode_completed\":true,\"outside_pixels\":%llu,\"outside_channel_differences\":%llu,\"changed_pixels\":%llu}\n",(unsigned long long)outside,(unsigned long long)diff,(unsigned long long)changed);
 if(diff)fail("pixels outside allowed MCU area changed");
}
int main(void) {
 int argc;LPWSTR *wideargs=CommandLineToArgvW(GetCommandLineW(),&argc);if(!wideargs)fail("command line");char **v=allocate((size_t)argc*sizeof(char*));
 for(int i=0;i<argc;i++){int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wideargs[i],-1,NULL,0,NULL,NULL);if(!n)fail("command UTF-8");v[i]=allocate(n);WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wideargs[i],-1,v[i],n,NULL,NULL);}LocalFree(wideargs);
 if(argc==4&&!strcmp(v[1],"encode")){manifest(v[3]);encode_patches(v[2]);}
 else if(argc==5&&!strcmp(v[1],"patch")){manifest(v[4]);patch_file(v[2],v[3]);}
 else if(argc==4&&!strcmp(v[1],"hash")){manifest(v[3]);struct jpeg_decompress_struct d;struct jpeg_error_mgr e;FILE *f=load(&d,&e,v[2],0);supported(&d);bounds(&d);jvirt_barray_ptr *a=jpeg_read_coefficients(&d);hash_coefficients(&d,a);jpeg_finish_decompress(&d);jpeg_destroy_decompress(&d);fclose(f);}
 else if(argc==6&&!strcmp(v[1],"pixels")){if(strcmp(v[5],"0")&&strcmp(v[5],"2"))fail("pixel margin");manifest(v[4]);pixels(v[2],v[3],atoi(v[5]));}
 else fail("usage: encode SOURCE MANIFEST | patch SOURCE OUTPUT MANIFEST | hash SOURCE MANIFEST | pixels SOURCE OUTPUT MANIFEST 0|2");
 return 0;
}
