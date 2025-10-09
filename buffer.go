package rmbg

import "sync"

type blurBufferPool struct {
	pool sync.Pool
}

func newBlurBufferPool() *blurBufferPool {
	return &blurBufferPool{
		pool: sync.Pool{
			New: func() any {
				return &blurBuffer{}
			},
		},
	}
}

type blurBuffer struct {
	tmp   []uint8
	hPass []uint8
}

func (p *blurBufferPool) get(size int) *blurBuffer {
	buf := p.pool.Get().(*blurBuffer)
	if cap(buf.tmp) < size {
		buf.tmp = make([]uint8, size)
		buf.hPass = make([]uint8, size)
	} else {
		buf.tmp = buf.tmp[:size]
		buf.hPass = buf.hPass[:size]
	}
	return buf
}

func (p *blurBufferPool) put(buf *blurBuffer) {
	p.pool.Put(buf)
}
